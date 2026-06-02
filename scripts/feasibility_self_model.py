"""Phase 1.2: Feasibility self-model.

Asks: can we predict the next-chunk prediction error ε from the policy's
*recent* commanded actions + realized state alone (no Pi0.5 hidden state)?

If yes → ε is task-predictable from observable proprioception, and we have a
strong baseline before adding hidden-state inputs in phase 2. If no → the
signal isn't generalizable purely from external observables and phase 2 is
necessary, not optional.

Setup:
  - Per timestep t in a trace, build a feature window from steps [t-K, t]:
      * last K=5 commanded actions          (5×7 = 35 dim)
      * last K=5 realized ee_pos deltas      (5×3 = 15 dim)
      * current gripper state                (2 dim)
      * current ee_pos                       (3 dim)
  - Target: mean ‖cmd[:3] − realized[:3]‖ over the NEXT H=10 steps (chunk).
    This is the policy's next-chunk prediction error. If we can predict
    this, we can metacognitively flag impending failure before it happens.
  - Train: leave-one-trace-out cross-validation (80 folds). For each fold,
    train on 79 traces, evaluate on the held-out trace.
  - Ablations: commanded-only, proprio-only (realized + gripper + ee_pos),
    both.

Acceptance gate: R² > 0.5 on held-out traces for the "both" variant. If
much lower, even the strongest external-features model can't predict ε,
and we need Pi0.5 internals.

Run:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/feasibility_self_model.py
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "self_model_phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5   # input window: last K commanded + last K realized
H = 10  # prediction horizon: mean ε over next H steps


def extract_features_for_trace(path: pathlib.Path) -> dict:
    d = np.load(path, allow_pickle=True)
    action = d["action"]                           # (T, 7)
    ee_pos = d["ee_pos"]                           # (T, 3)
    gripper = d["gripper_qpos"]                    # (T, 2)
    T = action.shape[0]
    if T < K + H + 2:
        return None
    cmd_xyz = action[:, :3]
    realized = np.zeros_like(cmd_xyz)
    realized[:-1] = np.diff(ee_pos, axis=0)        # (T-1, 3); last row is 0 padding

    # Targets: mean ‖cmd − realized‖ over next H steps (translation only)
    eps_per_step = np.linalg.norm(cmd_xyz - realized, axis=1)  # (T,)

    # Build per-timestep training rows for t in [K-1, T-H-1].
    feats_commanded = []
    feats_proprio = []
    feats_both = []
    targets = []
    for t in range(K - 1, T - H - 1):
        cmd_window = cmd_xyz[t - K + 1:t + 1].reshape(-1)   # K*3 commanded translations
        cmd_full_window = action[t - K + 1:t + 1].reshape(-1)  # K*7 (incl gripper)
        real_window = realized[t - K + 1:t + 1].reshape(-1)  # K*3 realized deltas
        grip_now = gripper[t]                                # (2,)
        ee_now = ee_pos[t]                                   # (3,)
        feats_commanded.append(cmd_full_window)
        feats_proprio.append(np.concatenate([real_window, grip_now, ee_now]))
        feats_both.append(np.concatenate([cmd_full_window, real_window, grip_now, ee_now]))
        targets.append(eps_per_step[t + 1:t + 1 + H].mean())

    return {
        "trace": path.name,
        "commanded": np.stack(feats_commanded),
        "proprio": np.stack(feats_proprio),
        "both": np.stack(feats_both),
        "target": np.array(targets),
    }


def loo_evaluate(features: list[dict], variant: str, hidden: tuple = (64, 32), seed: int = 0) -> dict:
    """Leave-one-trace-out evaluation. Returns per-trace R² + aggregate."""
    per_trace_r2 = []
    per_trace_pred_mean = []
    per_trace_true_mean = []
    rng = np.random.default_rng(seed)

    for i, held in enumerate(features):
        train = [f for j, f in enumerate(features) if j != i]
        X_train = np.concatenate([f[variant] for f in train], axis=0)
        y_train = np.concatenate([f["target"] for f in train], axis=0)
        # Subsample if too large (speed)
        if len(X_train) > 30000:
            idx = rng.choice(len(X_train), 30000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(held[variant])
        y_test = held["target"]

        model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=120,
                             random_state=seed, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=10)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        per_trace_r2.append(float(r2_score(y_test, y_pred)))
        per_trace_pred_mean.append(float(y_pred.mean()))
        per_trace_true_mean.append(float(y_test.mean()))

    per_trace_r2 = np.array(per_trace_r2)
    return {
        "r2_mean": float(per_trace_r2.mean()),
        "r2_median": float(np.median(per_trace_r2)),
        "r2_min": float(per_trace_r2.min()),
        "r2_p25": float(np.percentile(per_trace_r2, 25)),
        "r2_p75": float(np.percentile(per_trace_r2, 75)),
        "n_traces": int(len(per_trace_r2)),
        "n_traces_pos_r2": int((per_trace_r2 > 0).sum()),
        "n_traces_r2_above_0.5": int((per_trace_r2 > 0.5).sum()),
        "per_trace_r2": per_trace_r2.tolist(),
        "per_trace_pred_mean": per_trace_pred_mean,
        "per_trace_true_mean": per_trace_true_mean,
    }


def main() -> None:
    print(f"K={K}  H={H}")
    print(f"loading traces...")
    features = []
    skipped = 0
    for p in sorted(TRACES_DIR.glob("*.npz")):
        f = extract_features_for_trace(p)
        if f is None:
            skipped += 1
            continue
        features.append(f)
    print(f"got {len(features)} traces ({skipped} skipped for length)")
    print(f"feature dims: commanded={features[0]['commanded'].shape[1]}  "
          f"proprio={features[0]['proprio'].shape[1]}  both={features[0]['both'].shape[1]}")
    print(f"per-trace sample counts: median={np.median([len(f['target']) for f in features]):.0f}")

    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    label_of = {r["path"]: r["label"] for r in summary["rows"]}

    results = {}
    for variant in ["commanded", "proprio", "both"]:
        print(f"\n=== LOO evaluation: variant={variant} ===")
        res = loo_evaluate(features, variant)
        results[variant] = res
        print(f"  R² mean={res['r2_mean']:.3f}  median={res['r2_median']:.3f}  "
              f"p25={res['r2_p25']:.3f}  p75={res['r2_p75']:.3f}")
        print(f"  traces with R²>0:    {res['n_traces_pos_r2']}/{res['n_traces']}")
        print(f"  traces with R²>0.5:  {res['n_traces_r2_above_0.5']}/{res['n_traces']}")

    # Per-stratum predicted-ε vs true-ε (does the model correctly assign high ε to failures?)
    print("\n=== Per-stratum aggregate prediction ('both' variant) ===")
    by_stratum_true = defaultdict(list)
    by_stratum_pred = defaultdict(list)
    for tr, true_mean, pred_mean in zip(
        [f["trace"] for f in features],
        results["both"]["per_trace_true_mean"],
        results["both"]["per_trace_pred_mean"],
    ):
        lab = label_of.get(tr, "UNK")
        by_stratum_true[lab].append(true_mean)
        by_stratum_pred[lab].append(pred_mean)

    print(f"{'stratum':30s} {'N':>4s} {'true ε mean':>14s} {'pred ε mean':>14s}")
    for lab in sorted(by_stratum_true):
        tv = np.array(by_stratum_true[lab])
        pv = np.array(by_stratum_pred[lab])
        print(f"{lab:30s} {len(tv):>4d} {tv.mean():>14.4f} {pv.mean():>14.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, variant in zip(axes, ["commanded", "proprio", "both"]):
        r2s = results[variant]["per_trace_r2"]
        ax.hist(r2s, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(0.5, color="green", linestyle="--", alpha=0.5, label="R²=0.5 gate")
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_title(f"variant={variant}\nmean R²={results[variant]['r2_mean']:.2f}, median={results[variant]['r2_median']:.2f}")
        ax.set_xlabel("Per-trace R²")
        ax.set_ylabel("# traces")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loo_r2_distribution.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'loo_r2_distribution.png'}")

    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": {"K": K, "H": H, "hidden": [64, 32]},
        "results": results,
    }, indent=2))
    print(f"saved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
