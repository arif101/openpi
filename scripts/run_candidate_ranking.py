"""Rank LoRA improvement candidates via offline world-model evaluation.

For each candidate, we expect a rollout npz file produced by running
scripts/run_collect_rollouts.py against that candidate's checkpoint on a
fixed held-out set of LIBERO tasks (the held-out set MUST be the same
across candidates to make the scores comparable).

Reads:
  - World model checkpoint (data/contact_mpc/mpc_results/world_model.pt + _config.pt)
  - Value function checkpoint (data/contact_mpc/value_function/value_function.pt + _config.pt)
  - One or more candidate rollout npz files, each named by candidate

Writes:
  - ranked_candidates.json with per-candidate imagined_score + ranking

Usage:
    PYTHONPATH=src python3 scripts/run_candidate_ranking.py \
        --world-model data/contact_mpc/mpc_results/world_model.pt \
        --value-function data/contact_mpc/value_function/value_function.pt \
        --candidate-rollouts \
            planning_0:data/contact_mpc/candidates/planning_0/rollouts.npz \
            skill_0:data/contact_mpc/candidates/skill_0/rollouts.npz \
        --output data/contact_mpc/ranked_candidates.json
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import torch

from openpi.contact_mpc.eval.offline_evaluator import (
    rank_candidates,
    save_rankings,
    score_candidate_from_rollouts,
)
from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
from openpi.contact_mpc.world_model.architecture import LatentWorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-model", required=True,
                   help="Path to LatentWorldModel state_dict (.pt file).")
    p.add_argument("--world-model-config", default=None,
                   help="Path to WorldModelConfig (.pt). Defaults to <world-model>_config.pt.")
    p.add_argument("--value-function", required=True,
                   help="Path to PairwiseValueFunction state_dict (.pt file).")
    p.add_argument("--value-function-config", default=None,
                   help="Path to VF config dict. Defaults to <value-function>_config.pt.")
    p.add_argument("--candidate-rollouts", nargs="+", required=True,
                   help="Pairs like name:path/to/rollouts.npz")
    p.add_argument("--held-out-episode-ids", default=None,
                   help="Comma-separated episode IDs to score (optional filter).")
    p.add_argument("--output", default="data/contact_mpc/ranked_candidates.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def _default_config_path(model_path: str, suffix: str = "_config.pt") -> pathlib.Path:
    p = pathlib.Path(model_path)
    return p.parent / (p.stem + suffix)


def main():
    args = parse_args()
    device = torch.device(args.device)

    wm_config_path = pathlib.Path(
        args.world_model_config or _default_config_path(args.world_model)
    )
    wm_config = torch.load(wm_config_path, weights_only=False)
    world_model = LatentWorldModel(wm_config)
    world_model.load_state_dict(torch.load(args.world_model, weights_only=True))
    world_model = world_model.to(device).eval()
    logger.info(f"Loaded world model ({world_model.param_count():,} params) from {args.world_model}")

    vf_config_path = pathlib.Path(
        args.value_function_config or _default_config_path(args.value_function)
    )
    vf_config = torch.load(vf_config_path, weights_only=False)
    value_fn = PairwiseValueFunction(vf_config["input_dim"], vf_config["hidden_dim"])
    value_fn.load_state_dict(torch.load(args.value_function, weights_only=True))
    value_fn = value_fn.to(device).eval()
    logger.info(f"Loaded value function ({value_fn.param_count():,} params) from {args.value_function}")

    held_out: set[int] | None = None
    if args.held_out_episode_ids:
        held_out = {int(x) for x in args.held_out_episode_ids.split(",")}
        logger.info(f"Filtering to {len(held_out)} held-out episode ids")

    candidate_scores = []
    for spec in args.candidate_rollouts:
        if ":" not in spec:
            raise ValueError(f"Bad --candidate-rollouts entry '{spec}'. Expected name:path")
        name, path = spec.split(":", 1)
        score = score_candidate_from_rollouts(
            candidate_name=name,
            rollouts_npz_path=path,
            world_model=world_model,
            value_fn=value_fn,
            device=device,
            held_out_episode_ids=held_out,
            batch_size=args.batch_size,
        )
        candidate_scores.append(score)
        logger.info(
            f"  {name}: mean={score.imagined_score_mean:.4f} "
            f"std={score.imagined_score_std:.4f} "
            f"median={score.imagined_score_median:.4f} "
            f"N={score.num_observations}"
        )

    save_rankings(
        candidate_scores,
        pathlib.Path(args.output),
        extra={
            "world_model_path": args.world_model,
            "value_function_path": args.value_function,
            "held_out_episode_ids": sorted(held_out) if held_out else None,
        },
    )
    logger.info(f"Wrote rankings to {args.output}")

    logger.info("Ranking (best first):")
    for c in rank_candidates(candidate_scores):
        logger.info(f"  {c.candidate_name}: {c.imagined_score_mean:.4f}")


if __name__ == "__main__":
    main()
