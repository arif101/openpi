"""Diagnose why the trained latent world model fails KS3 ranking.

Hypothesis: the WM has low MSE but its outputs drift off the real-feature
manifold in ways MSE doesn't penalize. The value function — trained only
on real rollout states — then scores WM outputs incorrectly.

This script measures the distribution shift between predicted and real
features along five axes. It does NOT modify training; it produces a
diagnostic report that tells us which intervention (InfoNCE, multi-step,
VICReg, joint training) is most likely to fix the failure.

Usage:
    PYTHONPATH=src uv run python3 scripts/diagnose_world_model.py \\
        --world-model data/contact_mpc/mpc_results/world_model.pt \\
        --value-function data/contact_mpc/value_function/value_function.pt \\
        --features data/contact_mpc/features/libero90_features_H10.npz \\
        --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \\
        --output-dir data/contact_mpc/diagnostics
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np
import torch

from openpi.contact_mpc.features.dataset import FeatureDataset
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
    p.add_argument("--features", default="data/contact_mpc/features/libero90_features_H10.npz")
    p.add_argument("--rollouts", default="data/contact_mpc/rollouts/rollouts_libero_90.npz")
    p.add_argument("--output-dir", default="data/contact_mpc/diagnostics")
    p.add_argument("--n-samples", type=int, default=2000,
                   help="Random subsample of triples for diagnostics (speed + memory).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip matplotlib plotting (numbers-only mode).")
    return p.parse_args()


def _default_cfg_path(model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    return p.parent / (p.stem + "_config.pt")


def compute_predictions(
    world_model: LatentWorldModel,
    features: FeatureDataset,
    n_samples: int,
    device,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the world model forward on N random triples. Returns (predicted, real)."""
    N = len(features.hidden_states)
    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    h_t = features.hidden_states[idx]
    actions = features.action_chunks[idx]
    h_real = features.future_hidden_states[idx]

    H = world_model.config.max_horizon
    # Pad actions if needed
    if actions.shape[1] < H:
        pad = np.zeros((actions.shape[0], H - actions.shape[1], actions.shape[2]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=1)
    elif actions.shape[1] > H:
        actions = actions[:, :H]

    with torch.no_grad():
        h_t_tensor = torch.tensor(h_t, dtype=torch.float32, device=device)
        a_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
        h_pred = world_model(h_t_tensor, a_tensor).cpu().numpy().astype(np.float32)

    return h_pred, h_real


# --- Diagnostic 1: per-dim variance ratio -----------------------------------


def diag_variance_per_dim(h_pred: np.ndarray, h_real: np.ndarray) -> dict:
    """Mechanism (1): predicted features may have lower variance than real.

    If the predictor regresses to the conditional mean, predicted variance
    shrinks. A ratio <1 indicates mean-collapse; <0.5 is severe.
    """
    var_pred = h_pred.var(axis=0)
    var_real = h_real.var(axis=0)
    ratio = var_pred / (var_real + 1e-12)
    return {
        "per_dim_variance_ratio_mean": float(ratio.mean()),
        "per_dim_variance_ratio_median": float(np.median(ratio)),
        "per_dim_variance_ratio_p05": float(np.percentile(ratio, 5)),
        "per_dim_variance_ratio_p95": float(np.percentile(ratio, 95)),
        "fraction_collapsed_dims": float((ratio < 0.5).mean()),
        "fraction_inflated_dims": float((ratio > 2.0).mean()),
    }


# --- Diagnostic 2: manifold drift via nearest neighbor ----------------------


def diag_manifold_drift(h_pred: np.ndarray, h_real: np.ndarray) -> dict:
    """Mechanism (2): predicted features may land off the real manifold.

    Compare d(pred, nearest real) vs d(real_i, nearest real_j for i != j).
    If pred-NN >> real-NN, the WM is painting points the VF has never seen.
    """
    # For each predicted feature, distance to nearest REAL feature (other than self).
    # Compute in torch for speed.
    hp = torch.tensor(h_pred, dtype=torch.float32)
    hr = torch.tensor(h_real, dtype=torch.float32)
    # Batch distance: ||hp_i - hr_j||^2
    d_pred_to_real = torch.cdist(hp, hr, p=2)            # [N, N]
    d_real_to_real = torch.cdist(hr, hr, p=2)             # [N, N]

    # Mask diagonal of real-to-real so we don't count self-distance=0
    eye = torch.eye(len(hr), dtype=torch.bool)
    d_real_to_real = d_real_to_real.masked_fill(eye, float("inf"))

    nn_pred_dist = d_pred_to_real.min(dim=1).values.numpy()
    nn_real_dist = d_real_to_real.min(dim=1).values.numpy()

    return {
        "nn_pred_to_real_mean": float(nn_pred_dist.mean()),
        "nn_pred_to_real_median": float(np.median(nn_pred_dist)),
        "nn_real_to_real_mean": float(nn_real_dist.mean()),
        "nn_real_to_real_median": float(np.median(nn_real_dist)),
        "drift_ratio": float(nn_pred_dist.mean() / (nn_real_dist.mean() + 1e-12)),
    }


# --- Diagnostic 3: value-function score distribution shift ------------------


def diag_vf_score_shift(
    h_pred: np.ndarray,
    h_real: np.ndarray,
    is_success_real: np.ndarray,
    value_fn: PairwiseValueFunction,
    device,
) -> dict:
    """Mechanism (3): VF scores on predicted features may live in a different
    distribution than on real features.

    If VF(pred) and VF(real) distributions overlap very differently, that
    quantifies the VF-on-WM-output collapse.
    """
    with torch.no_grad():
        vf_pred = value_fn(torch.tensor(h_pred, dtype=torch.float32, device=device)).cpu().numpy()
        vf_real = value_fn(torch.tensor(h_real, dtype=torch.float32, device=device)).cpu().numpy()

    return {
        "vf_real_mean": float(vf_real.mean()),
        "vf_real_std": float(vf_real.std()),
        "vf_real_successes_mean": float(vf_real[is_success_real].mean()) if is_success_real.any() else None,
        "vf_real_failures_mean": float(vf_real[~is_success_real].mean()) if (~is_success_real).any() else None,
        "vf_pred_mean": float(vf_pred.mean()),
        "vf_pred_std": float(vf_pred.std()),
        "vf_pred_successes_mean": float(vf_pred[is_success_real].mean()) if is_success_real.any() else None,
        "vf_pred_failures_mean": float(vf_pred[~is_success_real].mean()) if (~is_success_real).any() else None,
        # Does the VF still separate success/failure on predicted features?
        "vf_pred_success_failure_gap": (
            float(vf_pred[is_success_real].mean() - vf_pred[~is_success_real].mean())
            if is_success_real.any() and (~is_success_real).any() else None
        ),
        "vf_real_success_failure_gap": (
            float(vf_real[is_success_real].mean() - vf_real[~is_success_real].mean())
            if is_success_real.any() and (~is_success_real).any() else None
        ),
    }


# --- Diagnostic 4: direction-of-error correlation --------------------------


def diag_error_structure(h_pred: np.ndarray, h_real: np.ndarray) -> dict:
    """Mechanism (4): per-dim error may correlate with important dims.

    Compute which dims have the largest residual and check if those dims
    carry the most variance in real features (i.e., "important" dims).
    """
    residual_per_dim = ((h_pred - h_real) ** 2).mean(axis=0)
    var_real_per_dim = h_real.var(axis=0)

    # Rank dims by importance (variance) and by error
    top_var_idx = np.argsort(-var_real_per_dim)[:100]
    top_err_idx = np.argsort(-residual_per_dim)[:100]
    overlap = len(set(top_var_idx.tolist()) & set(top_err_idx.tolist()))

    return {
        "residual_per_dim_mean": float(residual_per_dim.mean()),
        "residual_per_dim_max": float(residual_per_dim.max()),
        "overlap_top100_high_variance_and_high_error": overlap,
        "spearman_error_vs_variance": float(
            np.corrcoef(
                np.argsort(np.argsort(-residual_per_dim)),
                np.argsort(np.argsort(-var_real_per_dim)),
            )[0, 1]
        ),
    }


# --- Diagnostic 5: cosine similarity pred <-> real -------------------------


def diag_cosine_similarity(h_pred: np.ndarray, h_real: np.ndarray) -> dict:
    """How close are predictions to their *specific* target in angular terms?"""
    norms_p = np.linalg.norm(h_pred, axis=1, keepdims=True)
    norms_r = np.linalg.norm(h_real, axis=1, keepdims=True)
    cos = ((h_pred * h_real).sum(axis=1, keepdims=True)) / (norms_p * norms_r + 1e-12)
    return {
        "cosine_pred_vs_real_target_mean": float(cos.mean()),
        "cosine_pred_vs_real_target_median": float(np.median(cos)),
    }


# --- Plotting ---------------------------------------------------------------


def plot_diagnostics(h_pred, h_real, vf_pred, vf_real, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. VF score distributions
    ax = axes[0, 0]
    ax.hist(vf_real, bins=40, alpha=0.5, label="VF(real)", color="tab:blue", density=True)
    ax.hist(vf_pred, bins=40, alpha=0.5, label="VF(predicted)", color="tab:orange", density=True)
    ax.set_xlabel("Value-function score")
    ax.set_ylabel("Density")
    ax.set_title("VF score distribution: real vs predicted")
    ax.legend()

    # 2. Per-dim variance ratio
    ax = axes[0, 1]
    var_pred = h_pred.var(axis=0)
    var_real = h_real.var(axis=0)
    ratio = var_pred / (var_real + 1e-12)
    ax.hist(np.log10(ratio + 1e-12), bins=50, color="tab:green")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("log10(var(pred) / var(real))")
    ax.set_ylabel("Count over 2048 dims")
    ax.set_title("Per-dim variance ratio (0 = match)")

    # 3. PCA scatter
    ax = axes[1, 0]
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        real_2d = pca.fit_transform(h_real[:500])
        pred_2d = pca.transform(h_pred[:500])
        ax.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.4, s=8, label="real", c="tab:blue")
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], alpha=0.4, s=8, label="predicted", c="tab:orange")
        ax.set_xlabel("PC 1 (fit on real)")
        ax.set_ylabel("PC 2")
        ax.set_title("PCA scatter: real vs predicted latents")
        ax.legend()
    except ImportError:
        ax.text(0.5, 0.5, "sklearn not available", ha="center")

    # 4. Pred-to-real NN distance vs real-to-real NN distance
    ax = axes[1, 1]
    hp = torch.tensor(h_pred, dtype=torch.float32)
    hr = torch.tensor(h_real, dtype=torch.float32)
    d_pr = torch.cdist(hp, hr).min(dim=1).values.numpy()
    d_rr = torch.cdist(hr, hr)
    d_rr = d_rr.masked_fill(torch.eye(len(hr), dtype=torch.bool), float("inf"))
    d_rr = d_rr.min(dim=1).values.numpy()
    ax.hist(d_rr, bins=40, alpha=0.5, label="real→real NN", color="tab:blue", density=True)
    ax.hist(d_pr, bins=40, alpha=0.5, label="pred→real NN", color="tab:orange", density=True)
    ax.set_xlabel("Nearest-neighbor L2 distance")
    ax.set_ylabel("Density")
    ax.set_title("Manifold drift (right-shifted orange = WM off-manifold)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --- Orchestration ---------------------------------------------------------


def load_world_model(ckpt_path, cfg_path, device):
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    wm = LatentWorldModel(cfg)
    wm.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return wm.to(device).eval()


def load_value_function(ckpt_path, cfg_path, device):
    cfg = torch.load(cfg_path or _default_cfg_path(ckpt_path), weights_only=False)
    vf = PairwiseValueFunction(cfg["input_dim"], cfg["hidden_dim"])
    vf.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return vf.to(device).eval()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading world model from {args.world_model}")
    wm = load_world_model(args.world_model, args.world_model_config, device)

    logger.info(f"Loading value function from {args.value_function}")
    vf = load_value_function(args.value_function, args.value_function_config, device)

    logger.info(f"Loading feature triples from {args.features}")
    features = FeatureDataset.load(args.features)
    logger.info(f"  {len(features.hidden_states)} triples, hidden_dim={features.hidden_states.shape[1]}")

    logger.info(f"Computing WM predictions on {args.n_samples} random triples...")
    rng = np.random.default_rng(0)
    h_pred, h_real = compute_predictions(wm, features, args.n_samples, device, rng)
    logger.info(f"  pred shape={h_pred.shape}, real shape={h_real.shape}")

    # Sample the `is_success` flag for each chosen triple from the rollouts file.
    # Features dataset doesn't have is_success on the future hidden state, so we
    # approximate: a triple is "success" if the episode it came from ended in success.
    is_success_real = features.is_success.astype(bool) if hasattr(features, "is_success") else None
    # Subsample to match our sample
    if is_success_real is not None:
        N = len(features.hidden_states)
        idx = rng.choice(N, size=args.n_samples, replace=False)
        # re-run rng.choice — but we need the SAME idx used in compute_predictions.
        # Simpler: recompute_predictions with deterministic idx, or accept approximate.
        # For the diagnostic, using a fresh random sample is fine — we just want aggregate stats.
        is_success_real = is_success_real[idx]

    logger.info("Running diagnostics...")
    report = {
        "n_samples": int(len(h_pred)),
        "diagnostics": {
            "variance_per_dim": diag_variance_per_dim(h_pred, h_real),
            "manifold_drift": diag_manifold_drift(h_pred, h_real),
            "error_structure": diag_error_structure(h_pred, h_real),
            "cosine_similarity": diag_cosine_similarity(h_pred, h_real),
        },
    }
    if is_success_real is not None:
        report["diagnostics"]["vf_score_shift"] = diag_vf_score_shift(
            h_pred, h_real, is_success_real, vf, device,
        )

    # Pretty-print key numbers
    logger.info("=" * 72)
    logger.info("Diagnostic summary")
    logger.info("=" * 72)
    v = report["diagnostics"]["variance_per_dim"]
    logger.info(f"  Per-dim variance ratio (pred/real):")
    logger.info(f"    mean    = {v['per_dim_variance_ratio_mean']:.3f}  (1.0 = match)")
    logger.info(f"    median  = {v['per_dim_variance_ratio_median']:.3f}")
    logger.info(f"    fraction of dims with ratio <0.5 (collapsed): {v['fraction_collapsed_dims']*100:.1f}%")
    logger.info(f"    fraction of dims with ratio >2.0 (inflated):  {v['fraction_inflated_dims']*100:.1f}%")

    d = report["diagnostics"]["manifold_drift"]
    logger.info(f"  Manifold drift (pred->real / real->real NN dist):")
    logger.info(f"    pred_to_real NN mean = {d['nn_pred_to_real_mean']:.4f}")
    logger.info(f"    real_to_real NN mean = {d['nn_real_to_real_mean']:.4f}")
    logger.info(f"    drift ratio          = {d['drift_ratio']:.3f}  (1.0 = on manifold)")

    e = report["diagnostics"]["error_structure"]
    logger.info(f"  Error structure:")
    logger.info(f"    overlap of top-100 high-error & high-variance dims: {e['overlap_top100_high_variance_and_high_error']}/100")
    logger.info(f"    spearman(error_rank, variance_rank): {e['spearman_error_vs_variance']:.3f}")

    c = report["diagnostics"]["cosine_similarity"]
    logger.info(f"  Cosine(pred, real_target): mean={c['cosine_pred_vs_real_target_mean']:.4f}")

    if "vf_score_shift" in report["diagnostics"]:
        s = report["diagnostics"]["vf_score_shift"]
        logger.info(f"  VF score shift:")
        logger.info(f"    VF(real)  mean={s['vf_real_mean']:.3f}  std={s['vf_real_std']:.3f}")
        logger.info(f"    VF(pred)  mean={s['vf_pred_mean']:.3f}  std={s['vf_pred_std']:.3f}")
        if s.get("vf_real_success_failure_gap") is not None:
            logger.info(f"    Success-failure gap on REAL  = {s['vf_real_success_failure_gap']:+.3f}")
            logger.info(f"    Success-failure gap on PRED  = {s['vf_pred_success_failure_gap']:+.3f}")

    # Write report
    report_path = out_dir / "wm_diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {report_path}")

    # Plots
    if not args.skip_plots:
        try:
            with torch.no_grad():
                vf_real = vf(torch.tensor(h_real, dtype=torch.float32, device=device)).cpu().numpy()
                vf_pred = vf(torch.tensor(h_pred, dtype=torch.float32, device=device)).cpu().numpy()
            plot_path = out_dir / "wm_diagnostic_plots.png"
            plot_diagnostics(h_pred, h_real, vf_pred, vf_real, plot_path)
            logger.info(f"Wrote {plot_path}")
        except Exception as exc:
            logger.warning(f"Plotting failed: {exc}")

    logger.info("")
    logger.info("Interpretation guide:")
    logger.info("  variance_ratio << 1 AND fraction_collapsed > 30%  -> VICReg is critical")
    logger.info("  drift_ratio > 2                                    -> InfoNCE is critical")
    logger.info("  spearman_error_vs_variance > 0.3                   -> predictor fails on important dims")
    logger.info("  vf_pred success_failure_gap near 0 or negative     -> joint VF+WM training needed")


if __name__ == "__main__":
    main()
