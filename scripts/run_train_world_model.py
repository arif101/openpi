"""Day 3-5: Train latent world model on extracted VLM features.

Downloads features from HF (or loads from local), trains the world model
with size and horizon sweeps, evaluates kill switches.

Usage (on GPU machine):
    cd /workspace/openpi
    uv run python scripts/run_train_world_model.py \
        --features-path data/contact_mpc/features/libero90_features_H10.npz \
        --probe-path data/contact_mpc/features/best_probe.pkl \
        --output-dir data/contact_mpc/world_model \
        --horizons 3 5 10 \
        --sizes small medium large

    # Or download from HF first:
    uv run python scripts/run_train_world_model.py \
        --download-from-hf arif101/libero90_vlm_features \
        --output-dir data/contact_mpc/world_model
"""

import argparse
import logging
import pathlib
import pickle

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="data/contact_mpc/features/libero90_features_H10.npz")
    parser.add_argument("--probe-path", default="data/contact_mpc/features/best_probe.pkl")
    parser.add_argument("--download-from-hf", default=None, help="HF repo to download features from")
    parser.add_argument("--output-dir", default="data/contact_mpc/world_model")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-contact-only", action="store_true", help="Also train contact-only variant for KS2")
    parser.add_argument("--skip-sweep", action="store_true", help="Only train medium/H=10 (for debugging)")
    # Anti-collapse regularizers (default off for backwards compat)
    parser.add_argument("--use-infonce", action="store_true",
                        help="Add InfoNCE contrastive loss on predicted vs real. Attacks manifold drift.")
    parser.add_argument("--infonce-weight", type=float, default=0.1)
    parser.add_argument("--infonce-temperature", type=float, default=0.1)
    parser.add_argument("--use-vicreg", action="store_true",
                        help="Add VICReg regularization on predictor outputs. Attacks variance collapse.")
    parser.add_argument("--vicreg-variance-weight", type=float, default=1.0)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=0.04)
    parser.add_argument("--vicreg-target-std", type=float, default=1.0)
    parser.add_argument("--vicreg-match-real-std", action="store_true",
                        help="Compute target_std per-dim from real features. "
                             "Essential when encoder outputs aren't unit-std (our case).")
    return parser.parse_args()


def download_from_hf(repo_id: str, output_dir: str) -> tuple[str, str]:
    """Download features and probe from HuggingFace."""
    from huggingface_hub import hf_hub_download

    features_path = hf_hub_download(repo_id=repo_id, filename="libero90_features_H10.npz", repo_type="dataset")
    probe_path = hf_hub_download(repo_id=repo_id, filename="best_probe.pkl", repo_type="dataset")
    logger.info(f"Downloaded features from {repo_id}")
    return features_path, probe_path


def main():
    args = parse_args()

    from openpi.contact_mpc.features.dataset import FeatureDataset
    from openpi.contact_mpc.world_model.train import train_world_model, save_world_model
    from openpi.contact_mpc.world_model.evaluate import run_all_kill_switches

    # Load data
    if args.download_from_hf:
        features_path, probe_path = download_from_hf(args.download_from_hf, args.output_dir)
    else:
        features_path = args.features_path
        probe_path = args.probe_path

    features = FeatureDataset.load(features_path)
    logger.info(f"Loaded features: {features.hidden_states.shape[0]} triples, "
                f"hidden_dim={features.hidden_states.shape[1]}, H={features.horizon}")

    with open(probe_path, "rb") as f:
        proxy_scorer = pickle.load(f)
    logger.info(f"Loaded proxy scorer: {type(proxy_scorer).__name__}")

    output_dir = pathlib.Path(args.output_dir)

    # Determine sweep grid
    if args.skip_sweep:
        horizons = [10]
        sizes = ["medium"]
    else:
        horizons = args.horizons
        sizes = args.sizes

    # Train sweep
    all_results = {}
    best_model = None
    best_ks3 = 0.0
    best_tag = None

    for H in horizons:
        for size in sizes:
            tag = f"H{H}_{size}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Training: {tag}")
            logger.info(f"{'='*60}")

            # Train all-data model
            model, metrics = train_world_model(
                features, horizon=H, size=size,
                num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
                use_infonce=args.use_infonce,
                infonce_weight=args.infonce_weight,
                infonce_temperature=args.infonce_temperature,
                use_vicreg=args.use_vicreg,
                vicreg_variance_weight=args.vicreg_variance_weight,
                vicreg_covariance_weight=args.vicreg_covariance_weight,
                vicreg_target_std=args.vicreg_target_std,
                vicreg_match_real_std=args.vicreg_match_real_std,
            )
            save_world_model(model, metrics, str(output_dir))

            # Train contact-only model if requested
            contact_model = None
            if args.train_contact_only:
                try:
                    contact_model, contact_metrics = train_world_model(
                        features, horizon=H, size=size, contact_only=True,
                        num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
                    )
                    save_world_model(contact_model, contact_metrics, str(output_dir))
                except ValueError as e:
                    logger.warning(f"Contact-only training failed for {tag}: {e}")

            # Run kill switches
            ks_results = run_all_kill_switches(
                model, contact_model, features, H, proxy_scorer,
            )
            all_results[tag] = {"metrics": metrics, "kill_switches": ks_results}

            # Track best by KS3
            ks3_acc = ks_results["ks3"]["ranking_accuracy"]
            if ks3_acc > best_ks3:
                best_ks3 = ks3_acc
                best_model = model
                best_tag = tag

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("SWEEP SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Tag':<20} {'Params':>10} {'Val MSE':>10} {'KS1':>6} {'KS3 Rank':>10} {'KS3':>6}")
    logger.info("-" * 70)

    for tag, result in sorted(all_results.items()):
        m = result["metrics"]
        ks = result["kill_switches"]
        logger.info(f"{tag:<20} {m['param_count']:>10,} {m['val_loss_best']:>10.6f} "
                    f"{'PASS' if ks['ks1']['passes'] else 'FAIL':>6} "
                    f"{ks['ks3']['ranking_accuracy']:>10.3f} "
                    f"{'PASS' if ks['ks3']['passes'] else 'FAIL':>6}")

    # Selection rule: smallest model that passes KS3 at ≥70%
    passing_models = [(tag, r) for tag, r in all_results.items() if r["kill_switches"]["ks3"]["passes"]]
    if passing_models:
        # Sort by param count ascending
        passing_models.sort(key=lambda x: x[1]["metrics"]["param_count"])
        selected_tag = passing_models[0][0]
        logger.info(f"\nSELECTED: {selected_tag} (smallest model passing KS3 ≥70%)")
    else:
        logger.warning("\nNO MODEL PASSES KS3 ≥70%. Consider pivoting to R1 contingency (best-of-N value only).")
        if best_tag:
            logger.info(f"Best KS3: {best_tag} at {best_ks3:.3f}")

    # Save summary
    np.savez(
        output_dir / "sweep_summary.npz",
        tags=list(all_results.keys()),
        **{f"{tag}_ks3": r["kill_switches"]["ks3"]["ranking_accuracy"] for tag, r in all_results.items()},
        **{f"{tag}_val_mse": r["metrics"]["val_loss_best"] for tag, r in all_results.items()},
    )


if __name__ == "__main__":
    main()
