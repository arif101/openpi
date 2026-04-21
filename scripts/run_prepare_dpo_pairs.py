"""Build DPO training pairs per cluster from rollouts + taxonomy.

Reads:
  - clusters.json (from run_failure_clustering.py)
  - rollouts_libero_90.npz (from run_collect_rollouts.py)

Writes:
  - data/contact_mpc/candidates/<cluster_id>/dpo_pairs.npz

Each cluster gets its own pair set; we train one LoRA per cluster using
these pairs. Per-task proximity is enforced (a success and a failure
decision are paired only if their sim-timesteps are within a window).

Usage:
    PYTHONPATH=src uv run python3 scripts/run_prepare_dpo_pairs.py \
        --clusters data/contact_mpc/clusters.json \
        --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
        --output-root data/contact_mpc/candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

from openpi.contact_mpc.dpo.pair_builder import build_pairs_for_cluster, save_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default="data/contact_mpc/clusters.json")
    p.add_argument("--rollouts", default="data/contact_mpc/rollouts/rollouts_libero_90.npz")
    p.add_argument("--output-root", default="data/contact_mpc/candidates")
    p.add_argument("--n-pairs-per-task", type=int, default=50)
    p.add_argument("--timestep-window", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading clusters from {args.clusters}")
    clusters = json.loads(pathlib.Path(args.clusters).read_text())["clusters"]

    logger.info(f"Loading rollouts from {args.rollouts}")
    d = np.load(args.rollouts)
    rollouts_kwargs = {
        "hidden_states": d["hidden_states"],
        "action_chunks": d["action_chunks"],
        "episode_ids": d["episode_ids"],
        "task_ids": d["task_ids"],
        "timesteps": d["timesteps"],
        "is_success": d["is_success"].astype(bool),
    }

    summary = []
    for cluster in clusters:
        cid = cluster["cluster_id"]
        pairs = build_pairs_for_cluster(
            cluster_id=cid,
            cluster_task_ids=cluster["task_ids"],
            cluster_failure_episode_ids=cluster["episode_ids"],
            **rollouts_kwargs,
            n_pairs_per_task=args.n_pairs_per_task,
            timestep_window=args.timestep_window,
            seed=args.seed,
        )

        if not pairs:
            logger.warning(f"[{cid}] No pairs produced — skipping")
            summary.append({"cluster_id": cid, "num_pairs": 0})
            continue

        out_dir = pathlib.Path(args.output_root) / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "dpo_pairs.npz"
        save_pairs(pairs, str(out_path))

        # Quick diagnostic: how many unique tasks, episodes contributed
        unique_tasks = {p.task_id for p in pairs}
        unique_win = {p.preferred_episode_id for p in pairs}
        unique_lose = {p.rejected_episode_id for p in pairs}
        logger.info(
            f"[{cid}] {len(pairs)} pairs across {len(unique_tasks)} tasks "
            f"({len(unique_win)} win eps, {len(unique_lose)} lose eps) -> {out_path}"
        )
        summary.append({
            "cluster_id": cid,
            "num_pairs": len(pairs),
            "num_tasks": len(unique_tasks),
            "num_preferred_episodes": len(unique_win),
            "num_rejected_episodes": len(unique_lose),
            "output": str(out_path),
        })

    summary_path = pathlib.Path(args.output_root) / "dpo_pairs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
