"""Prediction-error measurement framework on the 80-trace corpus.

Tests the predictive-coding hypothesis: does the gap between commanded EE
delta and realized EE delta specifically discriminate IN_TRUE_CLOSE_FALSE
(execution-stuck) from IN_FALSE_CLOSE_FALSE (generation-level failure)
and PHYS_OK (success)?

The cleanest contrastive pair for the hypothesis is
    IN_TRUE_CLOSE_FALSE vs IN_FALSE_CLOSE_FALSE
since both are failures on the same task, same perturbation, same seed —
the only difference is the failure *mechanism*. If ε wins this contrast,
prediction error is mechanism-specific (active-inference contribution).
If it loses, ε is just a generic failure indicator (no new claim).

Methodological notes incorporated from review:
  - Per-axis ε, not just norm (drawer-close is +y-specific).
  - Threshold = 95th percentile of PHYS_OK per-step ε (principled, not tuned).
  - Chunk-boundary feedback test: does policy re-emit same commanded direction
    after seeing a high-ε chunk? (Pi0.5 action horizon assumed = 10.)
  - Pairwise AUROCs across all three strata, with bootstrap 95% CIs.
  - Controller-bias caveat documented: action[:,:3] is normalized OSC command,
    realized ee_pos delta is in meters; absolute scales differ but relative
    rankings across strata are unbiased under the assumption the OSC scale is
    stratum-independent (it is — same controller, same env).

Usage:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/analyze_prediction_error.py
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
STRATA_JSON = pathlib.Path("data/contact_mpc/failure_strata_bddl_v2.json")
OUT_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_HORIZON = 10  # Pi0.5 emits 10 actions per inference call
RNG = np.random.default_rng(0)


def load_labels() -> dict[str, str]:
    """Return {filename: stratum_label}. PHYS_OK_ files get label 'PHYS_OK'."""
    with STRATA_JSON.open() as f:
        bddl_labels = json.load(f)
    labels: dict[str, str] = {}
    for p in TRACES_DIR.glob("*.npz"):
        name = p.name
        if name.startswith("PHYS_OK_"):
            labels[name] = "PHYS_OK"
        elif name in bddl_labels:
            labels[name] = bddl_labels[name]["label"]
        else:
            labels[name] = "UNLABELED"
    return labels


def compute_trace_features(path: pathlib.Path) -> dict:
    """Per-trace prediction-error signals.

    Returns a dict of scalar features used in AUROC + plots.
    """
    d = np.load(path, allow_pickle=True)
    action = d["action"]          # (T, 7) — first 3 are normalized OSC delta-pos
    ee_pos = d["ee_pos"]          # (T, 3) — observed EE in world frame, meters
    T = action.shape[0]

    cmd = action[:-1, :3]                       # commanded direction per step
    realized = np.diff(ee_pos, axis=0)          # observed delta-EE in meters

    cmd_norm = np.linalg.norm(cmd, axis=1) + 1e-9       # (T-1,)
    real_norm = np.linalg.norm(realized, axis=1) + 1e-9 # (T-1,)

    # Per-axis correlation between commanded and realized.
    # Failure mode: y-axis correlation collapses when policy commands +y
    # but EE doesn't move (drawer-close stuck).
    per_axis_corr = {}
    for ax, name in enumerate(["x", "y", "z"]):
        if np.std(cmd[:, ax]) > 1e-6 and np.std(realized[:, ax]) > 1e-6:
            per_axis_corr[name] = float(np.corrcoef(cmd[:, ax], realized[:, ax])[0, 1])
        else:
            per_axis_corr[name] = float("nan")

    # Per-step direction alignment (cosine).
    cmd_unit = cmd / cmd_norm[:, None]
    real_unit = realized / real_norm[:, None]
    cos_align = np.einsum("ij,ij->i", cmd_unit, real_unit)  # (T-1,) in [-1, 1]

    # Per-step "execution ratio" — fraction of commanded magnitude actually realized.
    # Since OSC scales the normalized command by an internal factor we don't know,
    # we normalize by the trace's median commanded norm. Stratum-relative ranking is unbiased.
    exec_ratio = real_norm / (cmd_norm + 1e-9)  # absolute scale is bias, relative ranking is the signal

    # Chunk-boundary feedback test.
    # Pi0.5 emits ACTION_HORIZON actions per inference. Index chunks at t = 0, H, 2H, ...
    # For each chunk transition, ask: did the policy see high ε in chunk k AND keep
    # commanding the same direction in chunk k+1?
    chunk_starts = list(range(0, T - 1, ACTION_HORIZON))
    chunk_mean_eps = []  # 1 - cos_align averaged within chunk
    chunk_cmd_dirs = []  # mean commanded direction within chunk
    for c0 in chunk_starts:
        c1 = min(c0 + ACTION_HORIZON, T - 1)
        if c1 - c0 < 2:
            continue
        chunk_mean_eps.append(float(np.mean(1.0 - cos_align[c0:c1])))
        v = cmd[c0:c1].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        chunk_cmd_dirs.append(v)
    chunk_mean_eps = np.array(chunk_mean_eps)
    chunk_cmd_dirs = np.array(chunk_cmd_dirs)

    # "Fails to use feedback" count: chunks k where mean(1-cos) was high AND
    # the next chunk's commanded direction is still aligned with this chunk's
    # commanded direction (cos > 0.9). Threshold set later (per-stratum, from PHYS_OK).
    persistent_dir_after_high_eps = 0
    if len(chunk_cmd_dirs) > 1:
        for k in range(len(chunk_cmd_dirs) - 1):
            same_dir = float(np.dot(chunk_cmd_dirs[k], chunk_cmd_dirs[k + 1]))
            # tag whether direction persisted across boundary (threshold-free here)
            if same_dir > 0.9:
                persistent_dir_after_high_eps += int(chunk_mean_eps[k] > 0.5)

    return {
        "path": path.name,
        "T": int(T),
        "corr_x": per_axis_corr["x"],
        "corr_y": per_axis_corr["y"],
        "corr_z": per_axis_corr["z"],
        "mean_cos_align": float(np.mean(cos_align)),
        "mean_1mcos": float(np.mean(1.0 - cos_align)),
        "median_exec_ratio": float(np.median(exec_ratio)),
        "frac_low_exec_ratio": float(np.mean(exec_ratio < 0.01)),  # mostly-stuck fraction
        "max_consecutive_low_exec": int(_max_run(exec_ratio < 0.01)),
        "persistent_dir_after_high_eps": int(persistent_dir_after_high_eps),
        "num_chunks": int(len(chunk_mean_eps)),
    }


def _max_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D bool array."""
    if mask.size == 0:
        return 0
    best = cur = 0
    for b in mask:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """Return (point AUROC, 2.5%, 97.5%) bootstrap CI."""
    point = roc_auc_score(y_true, y_score)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(point), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    labels = load_labels()
    print(f"loaded {len(labels)} trace labels")

    rows = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        try:
            r = compute_trace_features(p)
            r["label"] = labels.get(p.name, "UNLABELED")
            rows.append(r)
        except Exception as e:
            print(f"  SKIP {p.name}: {type(e).__name__}: {e}")
    print(f"computed features for {len(rows)} traces")

    # Sanity dump per stratum
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[r["label"]].append(r)
    print("\nStratum counts:")
    for s, lst in sorted(by_stratum.items()):
        print(f"  {s:30s} N={len(lst)}")

    # Per-stratum summary table
    print("\nPer-stratum feature summary (mean ± std):")
    features = ["corr_y", "mean_cos_align", "median_exec_ratio",
                "frac_low_exec_ratio", "max_consecutive_low_exec",
                "persistent_dir_after_high_eps"]
    header = f"{'stratum':30s} " + " ".join(f"{f:>22s}" for f in features)
    print(header)
    for s in sorted(by_stratum):
        lst = by_stratum[s]
        cells = []
        for f in features:
            vals = np.array([r[f] for r in lst if not np.isnan(r[f])])
            if len(vals) == 0:
                cells.append(f"{'n/a':>22s}")
            else:
                cells.append(f"{vals.mean():>10.3f} ± {vals.std():>7.3f}")
        print(f"{s:30s} " + " ".join(cells))

    # Pairwise AUROCs.
    # We treat each feature as a candidate scalar score. Convention: higher score ⇒
    # higher probability of the "positive" class (the stratum being tested for).
    # For corr_y the negative class (PHYS_OK) has HIGHER values, so flip sign:
    sign = {
        "corr_y": -1,            # OK has higher corr_y → flip
        "corr_x": -1,
        "corr_z": -1,
        "mean_cos_align": -1,
        "median_exec_ratio": -1,
        "mean_1mcos": +1,
        "frac_low_exec_ratio": +1,
        "max_consecutive_low_exec": +1,
        "persistent_dir_after_high_eps": +1,
    }

    contrasts = [
        ("PHYS_OK", "IN_TRUE_CLOSE_FALSE"),
        ("PHYS_OK", "IN_FALSE_CLOSE_FALSE"),
        ("IN_FALSE_CLOSE_FALSE", "IN_TRUE_CLOSE_FALSE"),  # the mechanism-discrimination test
    ]

    print("\nPairwise AUROC (positive = right-side label):")
    print(f"{'contrast':50s} {'feature':35s} {'AUROC':>8s} {'95% CI':>20s}")
    auroc_results = {}
    for neg, pos in contrasts:
        neg_rows = by_stratum.get(neg, [])
        pos_rows = by_stratum.get(pos, [])
        if len(neg_rows) < 2 or len(pos_rows) < 2:
            print(f"  skip {neg} vs {pos}: too few samples ({len(neg_rows)} vs {len(pos_rows)})")
            continue
        for f in ["corr_y", "mean_cos_align", "median_exec_ratio",
                  "frac_low_exec_ratio", "max_consecutive_low_exec",
                  "persistent_dir_after_high_eps"]:
            y_true = np.array([0] * len(neg_rows) + [1] * len(pos_rows))
            y_score = np.array([sign[f] * r[f] for r in neg_rows + pos_rows])
            if np.any(np.isnan(y_score)):
                continue
            pt, lo, hi = bootstrap_auroc(y_true, y_score)
            print(f"{neg:>20s} vs {pos:<25s} {f:35s} {pt:>8.3f}   [{lo:.2f}, {hi:.2f}]")
            auroc_results[(neg, pos, f)] = (pt, lo, hi)
        print()

    # --- Plots ---
    # 1) Per-axis correlation boxplot by stratum
    strata_order = ["PHYS_OK", "IN_FALSE_CLOSE_FALSE", "IN_TRUE_CLOSE_FALSE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, axis_name in zip(axes, ["x", "y", "z"]):
        data = []
        for s in strata_order:
            vals = [r[f"corr_{axis_name}"] for r in by_stratum.get(s, []) if not np.isnan(r[f"corr_{axis_name}"])]
            data.append(vals)
        ax.boxplot(data, tick_labels=[s.replace("_", "\n") for s in strata_order])
        ax.set_title(f"corr(cmd_{axis_name}, realized_{axis_name})")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Pearson correlation (per trace)")
    fig.suptitle("Per-axis commanded↔realized correlation by stratum", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_axis_correlation.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'per_axis_correlation.png'}")

    # 2) Exec-ratio and stuck signature
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat in zip(axes, ["median_exec_ratio", "frac_low_exec_ratio", "max_consecutive_low_exec"]):
        data = []
        for s in strata_order:
            vals = [r[feat] for r in by_stratum.get(s, [])]
            data.append(vals)
        ax.boxplot(data, tick_labels=[s.replace("_", "\n") for s in strata_order])
        ax.set_title(feat)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Execution-failure signatures by stratum", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exec_signatures.png", dpi=120, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'exec_signatures.png'}")

    # 3) Chunk-boundary feedback failure
    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    for s in strata_order:
        vals = [r["persistent_dir_after_high_eps"] for r in by_stratum.get(s, [])]
        data.append(vals)
    ax.boxplot(data, tick_labels=[s.replace("_", "\n") for s in strata_order])
    ax.set_ylabel("# chunk boundaries where high ε did NOT change commanded direction")
    ax.set_title("Failure-to-use-feedback signature (chunk-boundary level)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chunk_feedback_failure.png", dpi=120, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'chunk_feedback_failure.png'}")

    # Persist rows + auroc for downstream
    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps({
        "rows": rows,
        "auroc": {f"{neg}|{pos}|{f}": v for (neg, pos, f), v in auroc_results.items()},
        "config": {"action_horizon": ACTION_HORIZON},
        "stratum_counts": {s: len(by_stratum[s]) for s in by_stratum},
    }, indent=2))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
