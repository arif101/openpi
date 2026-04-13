"""Train pairwise ranking probe and re-evaluate world model KS3.

The linear probe was too weak as a KS3 scorer (ground-truth ranking
accuracy only 0.661). This script trains an MLP with Bradley-Terry loss
directly optimized for pairwise ranking, then re-evaluates all saved
world models against the new scorer.

Usage:
    /workspace/openpi/.venv/bin/python scripts/run_ranking_probe.py \
        --world-model-dir data/contact_mpc/world_model
"""

import argparse
import logging
import pathlib
import pickle
import glob

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-hf-repo", default="arif101/libero90_vlm_features")
    parser.add_argument("--world-model-dir", default="data/contact_mpc/world_model")
    parser.add_argument("--output-dir", default="data/contact_mpc/ranking_probe")
    parser.add_argument("--n-train-pairs", type=int, default=50000)
    parser.add_argument("--num-epochs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    from huggingface_hub import hf_hub_download
    from openpi.contact_mpc.features.dataset import FeatureDataset
    from openpi.contact_mpc.features.ranking_probe import (
        RankingMLP,
        train_ranking_probe,
        evaluate_world_model_with_ranking_probe,
    )

    # Load features
    fp = hf_hub_download(args.features_hf_repo, "libero90_features_H10.npz", repo_type="dataset")
    features = FeatureDataset.load(fp)
    logger.info(f"Loaded {features.hidden_states.shape[0]} triples")

    # Train ranking probe
    logger.info("\n=== Training Ranking Probe ===")
    probe, val_acc = train_ranking_probe(
        features.hidden_states,
        features.episode_ids,
        features.timesteps,
        n_pairs=args.n_train_pairs,
        num_epochs=args.num_epochs,
    )
    logger.info(f"Ranking probe val accuracy: {val_acc:.3f}")

    # Diagnostic: ground-truth ranking accuracy with the new probe
    ep_lens = {}
    for i in range(len(features.episode_ids)):
        ep_lens.setdefault(int(features.episode_ids[i]), 0)
        ep_lens[int(features.episode_ids[i])] = max(ep_lens[int(features.episode_ids[i])], int(features.timesteps[i]) + 1)

    early = [i for i in range(len(features.timesteps)) if features.timesteps[i] / ep_lens[int(features.episode_ids[i])] < 0.5]
    late = [i for i in range(len(features.timesteps)) if features.timesteps[i] / ep_lens[int(features.episode_ids[i])] > 0.75]

    rng = np.random.RandomState(42)
    n = 1000
    ei, li = rng.choice(early, n, replace=True), rng.choice(late, n, replace=True)

    probe.eval()
    with torch.no_grad():
        raw_early = probe(torch.tensor(features.hidden_states[ei], dtype=torch.float32))
        raw_late = probe(torch.tensor(features.hidden_states[li], dtype=torch.float32))
        raw_acc = (raw_late > raw_early).float().mean().item()

        gt_early = probe(torch.tensor(features.future_hidden_states[ei], dtype=torch.float32))
        gt_late = probe(torch.tensor(features.future_hidden_states[li], dtype=torch.float32))
        gt_acc = (gt_late > gt_early).float().mean().item()

    logger.info(f"\nDiagnostic with ranking probe:")
    logger.info(f"  Raw h_t ranking:            {raw_acc:.3f}")
    logger.info(f"  Ground-truth h_{{t+H}} ranking: {gt_acc:.3f}")

    # Save probe
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), output_dir / "ranking_probe.pt")
    torch.save({"input_dim": features.hidden_states.shape[1], "hidden_dim": 256}, output_dir / "ranking_probe_config.pt")
    logger.info(f"Saved ranking probe to {output_dir}")

    # Re-evaluate all world models with the ranking probe
    wm_dir = pathlib.Path(args.world_model_dir)
    wm_files = sorted(wm_dir.glob("world_model_H*.pt"))
    wm_files = [f for f in wm_files if "config" not in f.name and "metrics" not in f.name]

    if wm_files:
        logger.info(f"\n=== Re-evaluating {len(wm_files)} world models with ranking probe ===")
        for wm_path in wm_files:
            tag = wm_path.stem.replace("world_model_", "")
            # Extract horizon from tag (e.g., "H10_medium" -> 10)
            H = int(tag.split("_")[0][1:])
            try:
                acc = evaluate_world_model_with_ranking_probe(
                    str(wm_path), fp, probe, horizon=H,
                )
                status = "PASS" if acc >= 0.70 else "FAIL"
                logger.info(f"  {tag}: KS3={acc:.3f} {status}")
            except Exception as e:
                logger.warning(f"  {tag}: error — {e}")
    else:
        logger.info("No world models found to re-evaluate.")


if __name__ == "__main__":
    main()
