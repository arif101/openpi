"""Cluster failure episodes by (failure_type × hidden-state similarity).

Reads:
  - failure_taxonomy.json (from run_vlm_attribution.py)
  - rollouts_libero_90.npz (from run_collect_rollouts.py) for hidden states

Writes:
  - clusters.json with K=3-5 interpretable cohorts for LoRA training

Usage:
    PYTHONPATH=src python3 scripts/run_failure_clustering.py \
        --taxonomy data/contact_mpc/failure_taxonomy.json \
        --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
        --output data/contact_mpc/clusters.json
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np

from openpi.contact_mpc.attribution.cluster import (
    cluster_failures,
    episode_level_features,
    load_attributions,
    save_clusters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taxonomy", default="data/contact_mpc/failure_taxonomy.json")
    p.add_argument("--rollouts", default="data/contact_mpc/rollouts/rollouts_libero_90.npz")
    p.add_argument("--output", default="data/contact_mpc/clusters.json")
    p.add_argument("--subcluster-threshold", type=int, default=80,
                   help="Primary failure_type groups larger than this get split by k-means.")
    p.add_argument("--max-subclusters", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading attributions from {args.taxonomy}")
    attributions = load_attributions(pathlib.Path(args.taxonomy))
    logger.info(f"  {len(attributions)} attributions")

    logger.info(f"Loading rollouts from {args.rollouts}")
    data = np.load(args.rollouts)
    hidden_states = data["hidden_states"]
    episode_ids = data["episode_ids"]

    features = episode_level_features(hidden_states, episode_ids)

    # Keep only episodes we have both attribution + features for
    common = set(attributions.keys()) & set(features.keys())
    missing_attr = set(features.keys()) - set(attributions.keys())
    missing_feat = set(attributions.keys()) - set(features.keys())
    if missing_feat:
        logger.warning(
            f"Skipping {len(missing_feat)} attributed episodes with no features "
            f"(first: {sorted(missing_feat)[:3]})"
        )
    if missing_attr:
        logger.info(f"Ignoring {len(missing_attr)} episodes without attribution (successes + unclassified).")

    attributions = {eid: attributions[eid] for eid in common}
    features = {eid: features[eid] for eid in common}

    logger.info(f"Clustering {len(attributions)} failure episodes...")
    clusters = cluster_failures(
        attributions,
        features,
        subcluster_threshold=args.subcluster_threshold,
        max_subclusters=args.max_subclusters,
        random_state=args.seed,
    )

    save_clusters(clusters, pathlib.Path(args.output))
    logger.info(f"Wrote {len(clusters)} clusters to {args.output}")

    for c in clusters:
        logger.info(
            f"  {c.cluster_id}: {len(c.episode_ids)} episodes "
            f"(tasks: {len(set(c.task_ids))}) — top cause: "
            f"{c.top_root_causes[0][:80] if c.top_root_causes else '-'}"
        )


if __name__ == "__main__":
    main()
