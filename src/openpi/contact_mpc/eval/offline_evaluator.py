"""Score LoRA improvement candidates against the latent world model.

Given (h_t, action_chunk) pairs produced by a candidate policy on a fixed
held-out set of initial observations, this module computes an
"imagined improvement score" — the mean value-function score of the
world-model-predicted future hidden states.

Scoring is deliberately split from rollout collection:
  - score_hidden_action_pairs is pure numpy/torch and unit-testable.
  - score_candidate_from_rollouts is the CLI-facing wrapper.

Rollouts for each candidate are collected separately (e.g., by invoking
run_collect_rollouts.py with --checkpoint pointing at each LoRA), and the
resulting rollouts_*.npz is fed into this evaluator.

The value function is known to memorize task identity (Phase 4 result),
so absolute scores are not meaningful. What we use is *relative ordering
across candidates on the same held-out observation set*. That is the
regime the evaluator is designed for.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np
import torch

from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
from openpi.contact_mpc.world_model.architecture import LatentWorldModel

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CandidateScore:
    """Aggregated imagined-improvement score for one LoRA candidate."""

    candidate_name: str
    num_observations: int
    imagined_score_mean: float
    imagined_score_std: float
    imagined_score_median: float
    per_observation_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "num_observations": self.num_observations,
            "imagined_score_mean": self.imagined_score_mean,
            "imagined_score_std": self.imagined_score_std,
            "imagined_score_median": self.imagined_score_median,
            "per_observation_scores": self.per_observation_scores,
        }


def _pad_or_trim_chunk(action_chunk: np.ndarray, target_H: int) -> np.ndarray:
    """Bring an action chunk to exactly target_H timesteps by zero-padding
    or truncation. Preserves action_dim."""
    H, d = action_chunk.shape
    if H == target_H:
        return action_chunk.astype(np.float32)
    if H > target_H:
        return action_chunk[:target_H].astype(np.float32)
    pad = np.zeros((target_H - H, d), dtype=np.float32)
    return np.concatenate([action_chunk.astype(np.float32), pad], axis=0)


def score_hidden_action_pairs(
    hidden_states: np.ndarray,
    action_chunks: np.ndarray,
    world_model: LatentWorldModel,
    value_fn: PairwiseValueFunction,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Score each (h_t, action_chunk) pair by world-model rollout + value function.

    Args:
        hidden_states: [N, hidden_dim] — current VLM features.
        action_chunks: [N, H, action_dim] — action chunk per observation. H can
            differ from world_model.config.max_horizon; will be padded/trimmed.
        world_model: trained LatentWorldModel.
        value_fn: trained PairwiseValueFunction.
        device: torch device to run on.
        batch_size: how many pairs to process per forward pass.

    Returns:
        scores: [N] numpy array of scalar imagined-improvement scores.

    Raises:
        ValueError if N mismatch between hidden_states and action_chunks.
    """
    if hidden_states.shape[0] != action_chunks.shape[0]:
        raise ValueError(
            f"N mismatch: hidden_states {hidden_states.shape[0]} vs "
            f"action_chunks {action_chunks.shape[0]}"
        )

    N = hidden_states.shape[0]
    target_H = world_model.config.max_horizon

    # Normalize action-chunk horizon
    padded = np.stack(
        [_pad_or_trim_chunk(action_chunks[i], target_H) for i in range(N)]
    ).astype(np.float32)

    world_model.eval()
    value_fn.eval()

    scores = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            h = torch.tensor(hidden_states[start:end], dtype=torch.float32, device=device)
            a = torch.tensor(padded[start:end], dtype=torch.float32, device=device)
            predicted = world_model(h, a)  # [batch, hidden_dim]
            batch_scores = value_fn(predicted)  # [batch]
            scores[start:end] = batch_scores.cpu().numpy().astype(np.float32)

    return scores


def aggregate_scores(candidate_name: str, scores: np.ndarray) -> CandidateScore:
    return CandidateScore(
        candidate_name=candidate_name,
        num_observations=int(scores.shape[0]),
        imagined_score_mean=float(np.mean(scores)),
        imagined_score_std=float(np.std(scores)),
        imagined_score_median=float(np.median(scores)),
        per_observation_scores=[float(s) for s in scores],
    )


def score_candidate_from_rollouts(
    candidate_name: str,
    rollouts_npz_path: str | pathlib.Path,
    world_model: LatentWorldModel,
    value_fn: PairwiseValueFunction,
    *,
    device: str | torch.device = "cpu",
    held_out_episode_ids: set[int] | None = None,
    batch_size: int = 32,
) -> CandidateScore:
    """Load a candidate's rollout npz and produce its aggregated score.

    If held_out_episode_ids is given, only decisions from those episodes
    are scored. This lets multiple candidates be scored on the same fixed
    held-out observation set even when candidates were collected
    independently.
    """
    data = np.load(rollouts_npz_path)
    hidden_states = data["hidden_states"]
    action_chunks = data["action_chunks"]
    episode_ids = data["episode_ids"]

    if held_out_episode_ids is not None:
        mask = np.isin(episode_ids, list(held_out_episode_ids))
        if not mask.any():
            raise ValueError(
                f"No decisions from held_out_episode_ids found in {rollouts_npz_path}"
            )
        hidden_states = hidden_states[mask]
        action_chunks = action_chunks[mask]

    scores = score_hidden_action_pairs(
        hidden_states=hidden_states,
        action_chunks=action_chunks,
        world_model=world_model,
        value_fn=value_fn,
        device=device,
        batch_size=batch_size,
    )
    return aggregate_scores(candidate_name, scores)


def score_candidate_per_task(
    candidate_name: str,
    rollouts_npz_path: str | pathlib.Path,
    world_model: LatentWorldModel,
    value_fn: PairwiseValueFunction,
    *,
    device: str | torch.device = "cpu",
    held_out_episode_ids: set[int] | None = None,
    batch_size: int = 32,
) -> dict[int, float]:
    """Return {task_id: mean_imagined_score} for one candidate.

    Used by the correlation study to pair imagined scores with real
    per-task success rates.
    """
    data = np.load(rollouts_npz_path)
    hidden_states = data["hidden_states"]
    action_chunks = data["action_chunks"]
    episode_ids = data["episode_ids"]
    task_ids = data["task_ids"]

    if held_out_episode_ids is not None:
        mask = np.isin(episode_ids, list(held_out_episode_ids))
        hidden_states = hidden_states[mask]
        action_chunks = action_chunks[mask]
        task_ids = task_ids[mask]

    if hidden_states.shape[0] == 0:
        return {}

    scores = score_hidden_action_pairs(
        hidden_states=hidden_states,
        action_chunks=action_chunks,
        world_model=world_model,
        value_fn=value_fn,
        device=device,
        batch_size=batch_size,
    )

    per_task: dict[int, float] = {}
    for tid in np.unique(task_ids):
        tid_scores = scores[task_ids == tid]
        per_task[int(tid)] = float(np.mean(tid_scores))
    return per_task


def rank_candidates(candidate_scores: list[CandidateScore]) -> list[CandidateScore]:
    """Return candidates sorted by imagined_score_mean descending."""
    return sorted(candidate_scores, key=lambda c: c.imagined_score_mean, reverse=True)


def save_rankings(
    candidate_scores: list[CandidateScore],
    path: pathlib.Path,
    extra: dict[str, Any] | None = None,
) -> None:
    ranked = rank_candidates(candidate_scores)
    payload: dict[str, Any] = {
        "num_candidates": len(ranked),
        "ranked_candidates": [c.to_dict() for c in ranked],
    }
    if extra:
        payload.update(extra)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(payload, indent=2))
