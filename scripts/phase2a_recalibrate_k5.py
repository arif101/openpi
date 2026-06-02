"""Recalibrate Phase 2a with k=5 hindsight (250ms controller-settling lookahead).

The k=1 streaming simulation showed online over-fires monotonically — partial
OSC responses register as low-exec. With k=5 hindsight, we wait 5 timesteps
(~250ms at 20Hz) before judging whether a command was realized.

Three sub-tasks:
  1. AUROC matrix (W × f) with k=5 — does the two-stage structure hold?
  2. Optimal thresholds at early gate (W=3, f=0.2) and late gate (W=5, f=0.5).
  3. Streaming simulation with k=5 + new threshold — does "first crossing" work now?

Run:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/phase2a_recalibrate_k5.py
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "phase2a_chunk_level_k5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01
LOOKAHEAD_K = 5
WINDOW_SIZES_CHUNKS = [2, 3, 5, 10]
FIRING_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def compute_per_step_low_exec_k(path: pathlib.Path, k: int = LOOKAHEAD_K) -> tuple[np.ndarray, int]:
    """realized[s] = ee_pos[s+k] - ee_pos[s]. Wait k steps before judging."""
    d = np.load(path, allow_pickle=True)
    action = d["action"]
    ee_pos = d["ee_pos"]
    T = action.shape[0]
    if T <= k:
        return None, T
    cmd_xyz = action[:T - k, :3]
    realized = ee_pos[k:k + (T - k)] - ee_pos[:T - k]
    cmd_norm = np.linalg.norm(cmd_xyz, axis=1) + 1e-9
    real_norm = np.linalg.norm(realized, axis=1)
    low_exec = (real_norm / cmd_norm < LOW_EXEC_THRESHOLD).astype(np.float32)
    # pad last k entries with 0 (unmeasurable at deployment time without that lookahead)
    low_exec_padded = np.concatenate([low_exec, np.zeros(k, dtype=np.float32)])
    return low_exec_padded, T


def main() -> None:
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    # Load all traces with k=5 low-exec
    traces = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        low_exec, T = compute_per_step_low_exec_k(p, k=LOOKAHEAD_K)
        if low_exec is None:
            continue
        if p.name.startswith("PHYS_OK_"):
            stratum = "PHYS_OK"
        elif p.name in cluster_of:
            stratum = f"C{cluster_of[p.name]}"
        else:
            continue
        traces.append({"name": p.name, "low_exec": low_exec, "T": T, "stratum": stratum})
    print(f"loaded {len(traces)} traces with k={LOOKAHEAD_K} hindsight")
    print(f"strata: " + ", ".join(f"{s}={sum(1 for t in traces if t['stratum']==s)}" for s in ["PHYS_OK", "C0", "C1"]))

    # (1) AUROC matrix with k=5
    print(f"\n=== AUROC: non-C1 vs C1 with k={LOOKAHEAD_K} hindsight ===")
    print(f"  {'f →':>10s}  " + "  ".join(f"f={f:.1f}" for f in FIRING_FRACTIONS))
    aurocs = {}
    for W in WINDOW_SIZES_CHUNKS:
        window_steps = W * ACTION_HORIZON
        row = f"  W={W:>2d} chunks "
        for f in FIRING_FRACTIONS:
            samples = []
            for t in traces:
                fire = int(f * t["T"])
                if fire < window_steps:
                    continue
                wf = t["low_exec"][fire - window_steps:fire].mean()
                samples.append((wf, t["stratum"]))
            neg = [v for v, st in samples if st != "C1"]
            pos = [v for v, st in samples if st == "C1"]
            if len(neg) < 2 or len(pos) < 2:
                row += f"  {'n/a':>10s}"
                continue
            y = np.array([0] * len(neg) + [1] * len(pos))
            s_arr = np.array(neg + pos)
            auc = roc_auc_score(y, s_arr)
            if auc < 0.5:
                auc = 1 - auc
            aurocs[(W, f)] = float(auc)
            row += f"  {auc:.2f} (N={len(samples)})"
        print(row)

    print(f"\n=== AUROC: C0 vs C1 with k={LOOKAHEAD_K} (mechanism contrast) ===")
    print(f"  {'f →':>10s}  " + "  ".join(f"f={f:.1f}" for f in FIRING_FRACTIONS))
    mech_aurocs = {}
    for W in WINDOW_SIZES_CHUNKS:
        window_steps = W * ACTION_HORIZON
        row = f"  W={W:>2d} chunks "
        for f in FIRING_FRACTIONS:
            samples = []
            for t in traces:
                fire = int(f * t["T"])
                if fire < window_steps:
                    continue
                wf = t["low_exec"][fire - window_steps:fire].mean()
                samples.append((wf, t["stratum"]))
            c0 = [v for v, st in samples if st == "C0"]
            c1 = [v for v, st in samples if st == "C1"]
            if len(c0) < 2 or len(c1) < 2:
                row += f"  {'n/a':>10s}"
                continue
            y = np.array([0] * len(c0) + [1] * len(c1))
            s_arr = np.array(c0 + c1)
            auc = roc_auc_score(y, s_arr)
            if auc < 0.5:
                auc = 1 - auc
            mech_aurocs[(W, f)] = float(auc)
            row += f"  {auc:.2f}"
        print(row)

    # (2) Optimal thresholds at early and late gate operating points
    print("\n=== Optimal thresholds at two operating points (k=5) ===")
    operating_points = [("EARLY (W=3, f=0.2)", 3, 0.2), ("LATE (W=5, f=0.5)", 5, 0.5)]
    chosen_thresholds = {}
    for label, W, f in operating_points:
        window_steps = W * ACTION_HORIZON
        samples = []
        for t in traces:
            fire = int(f * t["T"])
            if fire < window_steps:
                continue
            wf = t["low_exec"][fire - window_steps:fire].mean()
            samples.append((wf, t["stratum"]))
        neg = np.array([v for v, st in samples if st != "C1"])
        pos = np.array([v for v, st in samples if st == "C1"])
        if len(neg) < 2 or len(pos) < 2:
            continue
        print(f"\n  {label}")
        print(f"    non-C1: mean={neg.mean():.3f} std={neg.std():.3f} (N={len(neg)})")
        print(f"    C1:     mean={pos.mean():.3f} std={pos.std():.3f} (N={len(pos)})")
        y = np.array([0] * len(neg) + [1] * len(pos))
        s_arr = np.concatenate([neg, pos])
        fpr, tpr, thr = roc_curve(y, s_arr)
        j = tpr - fpr
        jb = j.argmax()
        print(f"    Youden-optimal threshold: {thr[jb]:.3f}  TPR={tpr[jb]:.3f}  FPR={fpr[jb]:.3f}")
        fpr_lo = np.where(fpr <= 0.10)[0]
        if len(fpr_lo) > 0:
            k_idx = fpr_lo[-1]
            print(f"    FPR≤0.10 threshold: {thr[k_idx]:.3f}  TPR={tpr[k_idx]:.3f}  FPR={fpr[k_idx]:.3f}")
            chosen_thresholds[label] = {"threshold_fpr_le_10": float(thr[k_idx]),
                                        "tpr": float(tpr[k_idx]), "fpr": float(fpr[k_idx])}
        chosen_thresholds.setdefault(label, {})["threshold_youden"] = float(thr[jb])
        chosen_thresholds[label]["tpr_youden"] = float(tpr[jb])
        chosen_thresholds[label]["fpr_youden"] = float(fpr[jb])

    # (3) Streaming simulation with k=5 + new early-gate threshold (FPR≤0.1)
    print("\n=== Streaming 'first crossing' simulation with k=5 ===")
    early_thr = chosen_thresholds["EARLY (W=3, f=0.2)"].get("threshold_fpr_le_10",
                  chosen_thresholds["EARLY (W=3, f=0.2)"]["threshold_youden"])
    print(f"Using early-gate threshold = {early_thr:.3f} (W=3, FPR≤0.1)")

    by_stratum = defaultdict(list)
    window_steps_3 = 3 * ACTION_HORIZON
    for tr in traces:
        # Find earliest step where windowed feature exceeds threshold
        first_fire = None
        for t in range(window_steps_3, tr["T"] - LOOKAHEAD_K):
            wf = tr["low_exec"][t - window_steps_3:t].mean()
            if wf > early_thr:
                first_fire = t
                break
        by_stratum[tr["stratum"]].append((first_fire, tr["T"]))

    for stratum in ["PHYS_OK", "C0", "C1"]:
        rows = by_stratum.get(stratum, [])
        fired = [r for r in rows if r[0] is not None]
        never = [r for r in rows if r[0] is None]
        print(f"\n  {stratum} (N={len(rows)}):")
        print(f"    fired: {len(fired)}, never_fired: {len(never)}")
        if fired:
            fs = np.array([r[0] for r in fired])
            Ts = np.array([r[1] for r in fired])
            frac = fs / Ts
            print(f"    fire step:    median={int(np.median(fs))}  p25={int(np.percentile(fs,25))}  p75={int(np.percentile(fs,75))}")
            print(f"    fire fraction: median={np.median(frac):.2f}  p25={np.percentile(frac,25):.2f}  p75={np.percentile(frac,75):.2f}")

    # TPR/FPR of streaming gate
    fires_per_stratum = {s: sum(1 for r in by_stratum[s] if r[0] is not None) for s in by_stratum}
    tpr_stream = fires_per_stratum.get("C1", 0) / max(1, len(by_stratum.get("C1", [])))
    non_c1_fires = fires_per_stratum.get("PHYS_OK", 0) + fires_per_stratum.get("C0", 0)
    non_c1_total = len(by_stratum.get("PHYS_OK", [])) + len(by_stratum.get("C0", []))
    fpr_stream = non_c1_fires / max(1, non_c1_total)
    print(f"\n  Streaming gate summary: TPR={tpr_stream:.3f} (C1 catch)  FPR={fpr_stream:.3f} (non-C1 false fire)")

    # (4) Visualization: AUROC heatmap for k=5
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    M = np.zeros((len(WINDOW_SIZES_CHUNKS), len(FIRING_FRACTIONS)))
    for i, W in enumerate(WINDOW_SIZES_CHUNKS):
        for j, f in enumerate(FIRING_FRACTIONS):
            v = aurocs.get((W, f))
            M[i, j] = v if v is not None else np.nan
    im = axes[0].imshow(M, aspect="auto", vmin=0.5, vmax=1.0, cmap="RdYlGn")
    axes[0].set_xticks(range(len(FIRING_FRACTIONS)))
    axes[0].set_xticklabels([f"{f:.1f}" for f in FIRING_FRACTIONS])
    axes[0].set_yticks(range(len(WINDOW_SIZES_CHUNKS)))
    axes[0].set_yticklabels([f"W={W}" for W in WINDOW_SIZES_CHUNKS])
    axes[0].set_xlabel("Firing fraction f")
    axes[0].set_ylabel("Window size (chunks)")
    axes[0].set_title(f"AUROC: non-C1 vs C1 with k={LOOKAHEAD_K} hindsight")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            axes[0].text(j, i, f"{v:.2f}" if not np.isnan(v) else "", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes[0], label="AUROC")

    # Mechanism contrast
    M2 = np.zeros((len(WINDOW_SIZES_CHUNKS), len(FIRING_FRACTIONS)))
    for i, W in enumerate(WINDOW_SIZES_CHUNKS):
        for j, f in enumerate(FIRING_FRACTIONS):
            v = mech_aurocs.get((W, f))
            M2[i, j] = v if v is not None else np.nan
    im2 = axes[1].imshow(M2, aspect="auto", vmin=0.5, vmax=1.0, cmap="RdYlGn")
    axes[1].set_xticks(range(len(FIRING_FRACTIONS)))
    axes[1].set_xticklabels([f"{f:.1f}" for f in FIRING_FRACTIONS])
    axes[1].set_yticks(range(len(WINDOW_SIZES_CHUNKS)))
    axes[1].set_yticklabels([f"W={W}" for W in WINDOW_SIZES_CHUNKS])
    axes[1].set_xlabel("Firing fraction f")
    axes[1].set_ylabel("Window size (chunks)")
    axes[1].set_title(f"AUROC: C0 vs C1 (mechanism contrast) with k={LOOKAHEAD_K}")
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            v = M2[i, j]
            axes[1].text(j, i, f"{v:.2f}" if not np.isnan(v) else "", ha="center", va="center", fontsize=9)
    plt.colorbar(im2, ax=axes[1], label="AUROC")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auroc_grid_k5.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'auroc_grid_k5.png'}")

    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": {"k": LOOKAHEAD_K, "LOW_EXEC_THRESHOLD": LOW_EXEC_THRESHOLD,
                   "ACTION_HORIZON": ACTION_HORIZON},
        "aurocs_nonC1_vs_C1": {f"W={W},f={f:.1f}": v for (W, f), v in aurocs.items()},
        "aurocs_C0_vs_C1": {f"W={W},f={f:.1f}": v for (W, f), v in mech_aurocs.items()},
        "chosen_thresholds": chosen_thresholds,
        "streaming_tpr": tpr_stream,
        "streaming_fpr": fpr_stream,
    }, indent=2))
    print(f"saved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
