"""Imagined-vs-real correlation study orchestrator.

Pairs per-(candidate, task) imagined scores (from the offline world-model
evaluator) with real LIBERO success rates (from run_libero_candidate_eval),
computes Pearson r + 95% bootstrap CI, and produces a scatter plot that
is the closing beat of the demo video.

Inputs:
  - World model + value function checkpoints (to compute imagined scores)
  - One rollout npz file per candidate (collected via run_collect_rollouts.py
    with that candidate's checkpoint, on the same held-out task set)
  - One real-eval JSON per candidate (from run_libero_candidate_eval.py)

Outputs:
  - correlation.json
  - scatter_plot.png

Usage (pairs are specified as candidate_name=rollouts_npz=real_json triples):
    PYTHONPATH=src python3 scripts/run_correlation_study.py \
        --world-model data/contact_mpc/mpc_results/world_model.pt \
        --value-function data/contact_mpc/value_function/value_function.pt \
        --candidate planning_0=data/contact_mpc/candidates/p0/rollouts.npz=data/contact_mpc/real_eval/p0.json \
        --candidate skill_0=data/contact_mpc/candidates/s0/rollouts.npz=data/contact_mpc/real_eval/s0.json \
        --output-json data/contact_mpc/correlation.json \
        --output-plot data/contact_mpc/scatter_plot.png
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import torch

from openpi.contact_mpc.eval.correlation_study import (
    compute_correlation,
    pair_by_candidate_and_task,
    save_correlation,
    save_scatter_plot,
)
from openpi.contact_mpc.eval.offline_evaluator import score_candidate_per_task
from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
from openpi.contact_mpc.world_model.architecture import LatentWorldModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--world-model", required=True)
    p.add_argument("--world-model-config", default=None)
    p.add_argument("--value-function", required=True)
    p.add_argument("--value-function-config", default=None)
    p.add_argument(
        "--candidate", action="append", required=True,
        help="Triple name=rollouts.npz=real_eval.json. Pass once per candidate.",
    )
    p.add_argument("--output-json", default="data/contact_mpc/correlation.json")
    p.add_argument("--output-plot", default="data/contact_mpc/scatter_plot.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _default_cfg(model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    return p.parent / (p.stem + "_config.pt")


def parse_candidate_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split("=")
    if len(parts) != 3:
        raise ValueError(
            f"Bad --candidate '{spec}'. Expected name=rollouts.npz=real_eval.json"
        )
    return parts[0], parts[1], parts[2]


def load_real_per_task(json_path: str) -> dict[int, float]:
    data = json.loads(pathlib.Path(json_path).read_text())
    return {int(tid): v["success_rate"] for tid, v in data["per_task"].items()}


def main():
    args = parse_args()
    device = torch.device(args.device)

    wm_config = torch.load(
        args.world_model_config or _default_cfg(args.world_model),
        weights_only=False,
    )
    world_model = LatentWorldModel(wm_config)
    world_model.load_state_dict(torch.load(args.world_model, weights_only=True))
    world_model = world_model.to(device).eval()
    logger.info("World model loaded")

    vf_cfg = torch.load(
        args.value_function_config or _default_cfg(args.value_function),
        weights_only=False,
    )
    value_fn = PairwiseValueFunction(vf_cfg["input_dim"], vf_cfg["hidden_dim"])
    value_fn.load_state_dict(torch.load(args.value_function, weights_only=True))
    value_fn = value_fn.to(device).eval()
    logger.info("Value function loaded")

    imagined: dict[str, dict[int, float]] = {}
    real: dict[str, dict[int, float]] = {}
    for spec in args.candidate:
        name, rollout_path, real_path = parse_candidate_spec(spec)
        logger.info(f"Scoring candidate {name}")
        imagined[name] = score_candidate_per_task(
            candidate_name=name,
            rollouts_npz_path=rollout_path,
            world_model=world_model,
            value_fn=value_fn,
            device=device,
        )
        real[name] = load_real_per_task(real_path)

    imagined_arr, real_arr, labels = pair_by_candidate_and_task(imagined, real)
    if len(imagined_arr) < 3:
        logger.error(
            f"Not enough aligned pairs: got {len(imagined_arr)}. "
            "Ensure candidates were evaluated on overlapping tasks."
        )
        return

    logger.info(f"Aligned {len(imagined_arr)} (candidate, task) pairs")

    result = compute_correlation(
        imagined=imagined_arr,
        real=real_arr,
        labels=labels,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    save_correlation(result, pathlib.Path(args.output_json))
    save_scatter_plot(result, pathlib.Path(args.output_plot))

    logger.info("=" * 60)
    logger.info(f"Pearson r = {result.pearson_r:.3f} "
                f"[{result.ci_low:.3f}, {result.ci_high:.3f}]  "
                f"(p = {result.pearson_p_value:.4f})")
    logger.info(f"Spearman r = {result.spearman_r:.3f}  "
                f"(p = {result.spearman_p_value:.4f})")
    logger.info(f"Pairs: {result.n_pairs}")
    logger.info(f"Strength: {result.strength}")
    logger.info(f"Wrote {args.output_json} + {args.output_plot}")


if __name__ == "__main__":
    main()
