"""Build (obs, action_preferred, action_rejected) pairs for DPO training.

Inputs:
  - A cluster from contact_mpc.attribution.cluster (names which episodes
    are failures on which tasks).
  - The full rollout FeatureDataset-compatible npz (hidden states + action
    chunks + episode/task IDs + is_success flags).

Output:
  - A list of DPOPair, one per pair. Pairs are constructed *within the
    same task* so the observation distribution is matched — a DPO gradient
    on such a pair isolates the action-quality signal.

Pair construction strategy:
  For each (task_id) in the cluster's matched task set:
    S = all success-episode decision points for that task
    F = all failure-episode decision points for that task (restricted to
        episodes actually in this cluster — not all task failures)
    Yield up to n_pairs_per_task random cross-joins of (s, f) from S x F
    where the decision timesteps are close (|t_s - t_f| <= timestep_window).

The timestep proximity constraint matters: comparing the last action of a
success to the first action of a failure is noise, not signal. We want
to compare *what the policy did at roughly the same phase of the task*.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DPOPair:
    """One preference-labeled training example.

    Fields are numpy arrays so the pair is framework-agnostic; JAX/Torch
    loaders tensorize at batch assembly time.
    """

    task_id: int
    cluster_id: str
    preferred_episode_id: int
    preferred_timestep: int
    preferred_hidden_state: np.ndarray  # [hidden_dim]
    preferred_action_chunk: np.ndarray  # [H, action_dim]
    rejected_episode_id: int
    rejected_timestep: int
    rejected_hidden_state: np.ndarray
    rejected_action_chunk: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": int(self.task_id),
            "cluster_id": self.cluster_id,
            "preferred_episode_id": int(self.preferred_episode_id),
            "preferred_timestep": int(self.preferred_timestep),
            "rejected_episode_id": int(self.rejected_episode_id),
            "rejected_timestep": int(self.rejected_timestep),
        }


def _task_decision_indices(
    task_ids: np.ndarray,
    is_success: np.ndarray,
    episode_ids: np.ndarray,
    task_id: int,
    *,
    want_success: bool,
    allowed_episode_ids: set[int] | None,
) -> np.ndarray:
    """Return decision-point indices for a given task filtered by is_success
    and optionally restricted to allowed_episode_ids."""
    mask = (task_ids == task_id) & (is_success == want_success)
    if allowed_episode_ids is not None:
        mask &= np.isin(episode_ids, list(allowed_episode_ids))
    return np.nonzero(mask)[0]


def build_pairs_for_cluster(
    *,
    cluster_id: str,
    cluster_task_ids: list[int],
    cluster_failure_episode_ids: list[int],
    hidden_states: np.ndarray,
    action_chunks: np.ndarray,
    episode_ids: np.ndarray,
    task_ids: np.ndarray,
    timesteps: np.ndarray,
    is_success: np.ndarray,
    n_pairs_per_task: int = 50,
    timestep_window: int = 15,
    seed: int = 0,
) -> list[DPOPair]:
    """Build DPO pairs for one cluster.

    Args:
        cluster_id: Identifier copied into each pair (useful for downstream filtering).
        cluster_task_ids: Task IDs this cluster's failures come from. Only these are paired.
        cluster_failure_episode_ids: Restrict failures to this cluster's members.
        hidden_states / action_chunks / episode_ids / task_ids / timesteps / is_success:
            Decision-point arrays from rollouts_libero_90.npz.
        n_pairs_per_task: Max pairs to sample per task.
        timestep_window: Only pair (s, f) if |t_s - t_f| <= this (in sim-timesteps).
        seed: RNG for pair sampling.

    Returns:
        List of DPOPair.
    """
    rng = np.random.default_rng(seed)
    allowed_failures = set(int(e) for e in cluster_failure_episode_ids)
    pairs: list[DPOPair] = []

    for task_id in cluster_task_ids:
        success_idx = _task_decision_indices(
            task_ids, is_success, episode_ids, task_id,
            want_success=True, allowed_episode_ids=None,
        )
        failure_idx = _task_decision_indices(
            task_ids, is_success, episode_ids, task_id,
            want_success=False, allowed_episode_ids=allowed_failures,
        )

        if len(success_idx) == 0 or len(failure_idx) == 0:
            logger.debug(
                f"[{cluster_id}] task {task_id}: "
                f"skipping ({len(success_idx)} successes, {len(failure_idx)} failures)"
            )
            continue

        s_times = timesteps[success_idx]
        f_times = timesteps[failure_idx]

        # Build every (s, f) pair whose timesteps are close, then subsample
        # up to n_pairs_per_task uniformly at random.
        # Use a simple O(|S| · |F|) filter — |S|, |F| are ~hundreds max per task.
        s_t_col = s_times[:, None]   # [|S|, 1]
        f_t_row = f_times[None, :]   # [1, |F|]
        close = np.abs(s_t_col - f_t_row) <= timestep_window  # [|S|, |F|]
        candidate_pairs = np.argwhere(close)  # [K, 2] of (s_pos, f_pos)

        if len(candidate_pairs) == 0:
            continue

        if len(candidate_pairs) > n_pairs_per_task:
            keep = rng.choice(len(candidate_pairs), size=n_pairs_per_task, replace=False)
            candidate_pairs = candidate_pairs[keep]

        for s_pos, f_pos in candidate_pairs:
            s_i = int(success_idx[s_pos])
            f_i = int(failure_idx[f_pos])
            pairs.append(
                DPOPair(
                    task_id=int(task_id),
                    cluster_id=cluster_id,
                    preferred_episode_id=int(episode_ids[s_i]),
                    preferred_timestep=int(timesteps[s_i]),
                    preferred_hidden_state=hidden_states[s_i].copy(),
                    preferred_action_chunk=action_chunks[s_i].copy(),
                    rejected_episode_id=int(episode_ids[f_i]),
                    rejected_timestep=int(timesteps[f_i]),
                    rejected_hidden_state=hidden_states[f_i].copy(),
                    rejected_action_chunk=action_chunks[f_i].copy(),
                )
            )

    return pairs


def save_pairs(pairs: list[DPOPair], npz_path: str) -> None:
    """Persist pairs as a single npz. Recoverable via load_pairs()."""
    if not pairs:
        raise ValueError("No pairs to save")

    np.savez(
        npz_path,
        task_ids=np.array([p.task_id for p in pairs], dtype=np.int64),
        cluster_ids=np.array([p.cluster_id for p in pairs]),
        preferred_episode_ids=np.array([p.preferred_episode_id for p in pairs], dtype=np.int64),
        preferred_timesteps=np.array([p.preferred_timestep for p in pairs], dtype=np.int64),
        preferred_hidden_states=np.stack([p.preferred_hidden_state for p in pairs]),
        preferred_action_chunks=np.stack([p.preferred_action_chunk for p in pairs]),
        rejected_episode_ids=np.array([p.rejected_episode_id for p in pairs], dtype=np.int64),
        rejected_timesteps=np.array([p.rejected_timestep for p in pairs], dtype=np.int64),
        rejected_hidden_states=np.stack([p.rejected_hidden_state for p in pairs]),
        rejected_action_chunks=np.stack([p.rejected_action_chunk for p in pairs]),
    )


def load_pairs(npz_path: str) -> list[DPOPair]:
    d = np.load(npz_path, allow_pickle=False)
    n = len(d["task_ids"])
    return [
        DPOPair(
            task_id=int(d["task_ids"][i]),
            cluster_id=str(d["cluster_ids"][i]),
            preferred_episode_id=int(d["preferred_episode_ids"][i]),
            preferred_timestep=int(d["preferred_timesteps"][i]),
            preferred_hidden_state=d["preferred_hidden_states"][i],
            preferred_action_chunk=d["preferred_action_chunks"][i],
            rejected_episode_id=int(d["rejected_episode_ids"][i]),
            rejected_timestep=int(d["rejected_timesteps"][i]),
            rejected_hidden_state=d["rejected_hidden_states"][i],
            rejected_action_chunk=d["rejected_action_chunks"][i],
        )
        for i in range(n)
    ]
