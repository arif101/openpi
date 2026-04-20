"""Prepare LoRA candidate training manifests from failure clusters.

For each cluster produced by `run_failure_clustering.py`, this script
identifies the matched success episodes (same tasks, but successful) and
writes a per-cluster manifest that drives LoRA training.

Actual LoRA training is an openpi `scripts/train.py` invocation (GPU-
bound, 2-4 hrs per candidate on an A40). This script does NOT launch
training — it produces the inputs and prints the commands.

Output per cluster:
  data/contact_mpc/candidates/<cluster_id>/
    manifest.json     # {task_ids, success_episode_ids, cluster_summary}
    README.md         # Suggested training command + rationale

The training command template uses the existing `pi05_libero_waypoint_lora`
config. If you need a per-cluster dataset subset, see the manifest's
success_episode_ids and either (a) add a filter hook to
LeRobotLiberoWaypointDataConfig, or (b) materialize a filtered LeRobot
dataset for each cluster.

Usage:
    PYTHONPATH=src python3 scripts/prepare_lora_candidates.py \
        --clusters data/contact_mpc/clusters.json \
        --success-rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
        --output-root data/contact_mpc/candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default="data/contact_mpc/clusters.json")
    p.add_argument(
        "--success-rollouts",
        default="data/contact_mpc/rollouts/rollouts_libero_90.npz",
        help=(
            "Rollouts npz containing both successes and failures; we filter to "
            "is_success==True to identify matched success episodes."
        ),
    )
    p.add_argument("--output-root", default="data/contact_mpc/candidates")
    p.add_argument("--config-name", default="pi05_libero_waypoint_lora",
                   help="openpi train config to use as baseline.")
    p.add_argument("--waypoint-dataset", default="arif101/libero90_waypoints",
                   help="HF LeRobot dataset with waypoint column.")
    return p.parse_args()


def load_success_episodes_per_task(rollouts_path: pathlib.Path) -> dict[int, list[int]]:
    """Return {task_id: [episode_ids]} for successful episodes only."""
    data = np.load(rollouts_path)
    episode_ids = data["episode_ids"]
    task_ids = data["task_ids"]
    is_success = data["is_success"]

    success_mask = is_success.astype(bool)
    unique_eids = set()
    task_to_episodes: dict[int, list[int]] = {}
    for i in range(len(episode_ids)):
        if not success_mask[i]:
            continue
        eid = int(episode_ids[i])
        tid = int(task_ids[i])
        if eid in unique_eids:
            continue
        unique_eids.add(eid)
        task_to_episodes.setdefault(tid, []).append(eid)
    return task_to_episodes


def write_manifest(
    cluster: dict,
    matched_success_episodes: list[int],
    output_dir: pathlib.Path,
    config_name: str,
    waypoint_dataset: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cluster_id": cluster["cluster_id"],
        "failure_type": cluster["failure_type"],
        "num_failure_episodes": cluster["num_episodes"],
        "cluster_task_ids": cluster["task_ids"],
        "matched_success_episode_ids": matched_success_episodes,
        "num_matched_successes": len(matched_success_episodes),
        "top_root_causes": cluster["top_root_causes"],
        "training_config": config_name,
        "waypoint_dataset": waypoint_dataset,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    exp_name = f"lora_{cluster['cluster_id']}"
    readme = f"""# LoRA candidate: {cluster['cluster_id']}

**Failure type:** {cluster['failure_type']}
**Targets {cluster['num_episodes']} failure episodes across {len(cluster['task_ids'])} tasks.**

## Top root causes in this cluster
{chr(10).join('- ' + rc for rc in cluster['top_root_causes'])}

## Training data

Train on the {len(matched_success_episodes)} matched-task success episodes:
```
task_ids:  {cluster['task_ids']}
episode_ids (from rollouts_libero_90.npz): {matched_success_episodes}
```

## Suggested training command

```bash
PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \\
/workspace/openpi/.venv/bin/python -u scripts/train.py \\
    {config_name} \\
    --exp-name={exp_name} \\
    --overwrite \\
    --num-train-steps=2000 \\
    --batch-size=8
```

**Note:** The existing `{config_name}` config trains on the full
`{waypoint_dataset}` dataset. To restrict to only this cluster's
matched-task successes, either:

  (a) Add a `task_ids` / `episode_ids` filter to
      `LeRobotLiberoWaypointDataConfig` in `src/openpi/training/config.py`,
      then pass the filter via a config override; or

  (b) Materialize a filtered LeRobot dataset (e.g.
      `arif101/libero90_waypoints_{cluster['cluster_id']}`) by filtering
      `{waypoint_dataset}` to `task_index in {cluster['task_ids']}` and
      republishing, then create a parallel training config pointing at
      the subset.

Path (b) is cleaner for the demo; 5 small HF dataset pushes is cheap and
gives the training pipeline zero new abstractions.

## Checkpoint output

After training, the LoRA checkpoint lives at:
```
data/checkpoints/{config_name}/{exp_name}/
```

Point downstream steps (`run_collect_rollouts.py`,
`run_libero_candidate_eval.py`) at that path via `--checkpoint`.
"""
    (output_dir / "README.md").write_text(readme)


def main():
    args = parse_args()
    logger.info(f"Loading clusters from {args.clusters}")
    clusters_data = json.loads(pathlib.Path(args.clusters).read_text())
    clusters = clusters_data["clusters"]
    logger.info(f"  {len(clusters)} clusters")

    logger.info(f"Loading success episodes from {args.success_rollouts}")
    success_per_task = load_success_episodes_per_task(pathlib.Path(args.success_rollouts))
    total_successes = sum(len(v) for v in success_per_task.values())
    logger.info(f"  {total_successes} successful episodes across {len(success_per_task)} tasks")

    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for cluster in clusters:
        matched: list[int] = []
        for tid in cluster["task_ids"]:
            matched.extend(success_per_task.get(int(tid), []))
        matched.sort()

        candidate_dir = output_root / cluster["cluster_id"]
        write_manifest(
            cluster=cluster,
            matched_success_episodes=matched,
            output_dir=candidate_dir,
            config_name=args.config_name,
            waypoint_dataset=args.waypoint_dataset,
        )
        logger.info(
            f"  {cluster['cluster_id']}: {cluster['num_episodes']} failures, "
            f"{len(matched)} matched successes → {candidate_dir}"
        )
        summary.append(
            {
                "cluster_id": cluster["cluster_id"],
                "num_failures": cluster["num_episodes"],
                "num_matched_successes": len(matched),
                "candidate_dir": str(candidate_dir),
            }
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {output_root / 'summary.json'}")

    # Also emit a consolidated training-orchestration script
    orchestrator = output_root / "train_all_candidates.sh"
    orchestrator.write_text(
        "#!/bin/bash\n"
        "# Train all LoRA candidates sequentially. Est. 3-4 hrs each on A40.\n"
        "set -euo pipefail\n\n"
        + "\n".join(
            f"echo '=== Training {c['cluster_id']} ==='\n"
            f"cat {output_root / c['cluster_id'] / 'README.md'} | grep -A 10 'Suggested training'\n"
            for c in summary
        )
        + "\n"
    )
    logger.info(f"Orchestrator stub: {orchestrator}")


if __name__ == "__main__":
    main()
