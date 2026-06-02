"""Phase 2a: chunk-level (runtime) feasibility of the frac_low_exec_ratio signal.

Phase 1 showed per-trace frac_low_exec_ratio cleanly separates C0/C1 in offline
analysis. Phase 2a asks: at chunk boundary k DURING a rollout, can we compute
a WINDOWED version of this signal that predicts eventual mechanism class
(C1 vs not)?

If yes (AUROC ≥ 0.85 with small W and early k) → real runtime gate.
If late-only (high AUROC only after t > 0.8T) → narrow intervention window.
If no → offline-only signal; the architectural claim weakens.

**Deliberately task-agnostic design:**
  - Uses ONLY frac_low_exec_ratio (the single load-bearing feature from Phase 1).
  - No per-axis features (corr_x/y/z would bake in task-3's +y dominance).
  - No commanded-action-history beyond the exec ratio.
  - 1% "low-exec" cutoff is the only OSC-specific calibration. Worth re-checking
    sensitivity (sweep at end).

The aim is to demonstrate the signal is mechanism-defined, not task-defined.
Phase 2b will then test whether the same THRESHOLD generalizes across tasks.

Run:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/phase2a_chunk_level_signal.py
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "phase2a_chunk_level"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01   # realized_norm < this × commanded_norm → "low-exec step"
WINDOW_SIZES_CHUNKS = [2, 3, 5, 10]   # try multiple windows
FIRING_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # what fraction of trace has elapsed when we fire


def compute_per_step_low_exec(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Return (low_exec_mask, T)."""
    d = np.load(path, allow_pickle=True)
    action = d["action"]              # (T, 7)
    ee_pos = d["ee_pos"]              # (T, 3)
    T = action.shape[0]
    cmd_xyz = action[:-1, :3]
    realized = np.diff(ee_pos, axis=0)
    cmd_norm = np.linalg.norm(cmd_xyz, axis=1) + 1e-9
    real_norm = np.linalg.norm(realized, axis=1)
    exec_ratio = real_norm / cmd_norm
    low_exec = (exec_ratio < LOW_EXEC_THRESHOLD).astype(np.float32)
    # pad to length T (last step has no realized delta)
    return np.concatenate([low_exec, [0.0]]), T


def main() -> None:
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    traces = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        low_exec_per_step, T = compute_per_step_low_exec(p)
        if p.name.startswith("PHYS_OK_"):
            stratum = "PHYS_OK"
        elif p.name in cluster_of:
            stratum = f"C{cluster_of[p.name]}"
        else:
            stratum = "UNK"
        traces.append({"name": p.name, "low_exec": low_exec_per_step, "T": T, "stratum": stratum})

    print(f"loaded {len(traces)} traces  (stratum counts: " + ", ".join(
        f"{s}={sum(1 for t in traces if t['stratum']==s)}"
        for s in ["PHYS_OK", "C0", "C1"]) + ")")

    # For each window size W chunks, and each firing fraction f, compute the
    # windowed frac_low_exec at the firing point, and check whether it
    # discriminates eventual stratum.
    print(f"\nLOW_EXEC_THRESHOLD = {LOW_EXEC_THRESHOLD}")
    print(f"Testing windows (chunks): {WINDOW_SIZES_CHUNKS}")
    print(f"Firing fractions of T: {FIRING_FRACTIONS}\n")

    aurocs = {}  # (W, f) → {contrast: auroc}
    fire_steps = {}  # for each (W, f, trace) → fire step in trace
    valid_samples_count = {}  # (W, f) → N

    for W in WINDOW_SIZES_CHUNKS:
        window_steps = W * ACTION_HORIZON
        for f in FIRING_FRACTIONS:
            samples = []  # (windowed_frac, stratum)
            for t in traces:
                fire_step = int(f * t["T"])
                if fire_step < window_steps:
                    continue   # not enough history
                w_start = fire_step - window_steps
                w_end = fire_step
                windowed_frac = t["low_exec"][w_start:w_end].mean()
                samples.append((windowed_frac, t["stratum"]))
            if not samples:
                continue
            valid_samples_count[(W, f)] = len(samples)

            # Contrasts
            row = {}
            for neg, pos in [("PHYS_OK", "C1"), ("non-C1", "C1"),
                             ("PHYS_OK", "C0"), ("C0", "C1")]:
                if neg == "non-C1":
                    neg_samples = [s for s, st in samples if st != "C1"]
                else:
                    neg_samples = [s for s, st in samples if st == neg]
                pos_samples = [s for s, st in samples if st == pos]
                if len(neg_samples) < 2 or len(pos_samples) < 2:
                    continue
                y = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
                s_arr = np.array(neg_samples + pos_samples)
                auc = roc_auc_score(y, s_arr)
                if auc < 0.5:
                    auc = 1 - auc
                row[f"{neg}_vs_{pos}"] = float(auc)
            aurocs[(W, f)] = row

    # Print a table for the most actionable contrast: PHYS_OK vs C1
    # (= "is this rollout about to be execution-stuck")
    print("=== AUROC: PHYS_OK vs C1 — 'will this rollout end execution-stuck?' ===")
    print(f"  {'f →':>10s}  " + "  ".join(f"f={f:.1f}" for f in FIRING_FRACTIONS))
    for W in WINDOW_SIZES_CHUNKS:
        row = f"  W={W:>2d} chunks "
        for f in FIRING_FRACTIONS:
            auc = aurocs.get((W, f), {}).get("PHYS_OK_vs_C1")
            n = valid_samples_count.get((W, f), 0)
            row += f"  {auc:.2f} (N={n})" if auc is not None else f"  {'n/a':>10s}"
        print(row)

    print("\n=== AUROC: non-C1 vs C1 — most decision-relevant contrast (binary gate) ===")
    print(f"  {'f →':>10s}  " + "  ".join(f"f={f:.1f}" for f in FIRING_FRACTIONS))
    for W in WINDOW_SIZES_CHUNKS:
        row = f"  W={W:>2d} chunks "
        for f in FIRING_FRACTIONS:
            auc = aurocs.get((W, f), {}).get("non-C1_vs_C1")
            n = valid_samples_count.get((W, f), 0)
            row += f"  {auc:.2f} (N={n})" if auc is not None else f"  {'n/a':>10s}"
        print(row)

    print("\n=== AUROC: C0 vs C1 — mechanism contrast (does the gate distinguish failure modes?) ===")
    print(f"  {'f →':>10s}  " + "  ".join(f"f={f:.1f}" for f in FIRING_FRACTIONS))
    for W in WINDOW_SIZES_CHUNKS:
        row = f"  W={W:>2d} chunks "
        for f in FIRING_FRACTIONS:
            auc = aurocs.get((W, f), {}).get("C0_vs_C1")
            n = valid_samples_count.get((W, f), 0)
            row += f"  {auc:.2f} (N={n})" if auc is not None else f"  {'n/a':>10s}"
        print(row)

    # Earliest firing time at which non-C1 vs C1 AUROC ≥ 0.85
    print("\n=== Earliest firing fraction f at which non-C1 vs C1 AUROC ≥ 0.85 ===")
    for W in WINDOW_SIZES_CHUNKS:
        earliest = None
        for f in FIRING_FRACTIONS:
            auc = aurocs.get((W, f), {}).get("non-C1_vs_C1")
            if auc is not None and auc >= 0.85:
                earliest = f
                break
        print(f"  W={W} chunks: earliest f = {earliest if earliest else 'never reaches 0.85'}")

    # Sensitivity to low_exec threshold (sweep)
    print("\n=== Sensitivity to LOW_EXEC_THRESHOLD (sweep at W=5, f=0.5) ===")
    print("Tests whether the 1% cutoff is load-bearing.")
    for thresh in [0.005, 0.01, 0.02, 0.05, 0.1]:
        samples = []
        for t in traces:
            d = np.load(TRACES_DIR / t["name"], allow_pickle=True)
            action = d["action"]
            ee_pos = d["ee_pos"]
            cmd_xyz = action[:-1, :3]
            realized = np.diff(ee_pos, axis=0)
            cmd_norm = np.linalg.norm(cmd_xyz, axis=1) + 1e-9
            real_norm = np.linalg.norm(realized, axis=1)
            ratio = real_norm / cmd_norm
            low = (ratio < thresh).astype(np.float32)
            low = np.concatenate([low, [0.0]])
            T = t["T"]
            fire_step = int(0.5 * T)
            if fire_step < 50:
                continue
            wf = low[max(0, fire_step - 50):fire_step].mean()
            samples.append((wf, t["stratum"]))
        # non-C1 vs C1 AUROC
        neg = [s for s, st in samples if st != "C1"]
        pos = [s for s, st in samples if st == "C1"]
        y = np.array([0] * len(neg) + [1] * len(pos))
        scores = np.array(neg + pos)
        auc = roc_auc_score(y, scores)
        if auc < 0.5:
            auc = 1 - auc
        print(f"  threshold={thresh:.3f}:  non-C1 vs C1 AUROC = {auc:.3f}")

    # Visualization: AUROC heatmap (W × f) for non-C1 vs C1
    fig, ax = plt.subplots(figsize=(8, 4))
    M = np.zeros((len(WINDOW_SIZES_CHUNKS), len(FIRING_FRACTIONS)))
    for i, W in enumerate(WINDOW_SIZES_CHUNKS):
        for j, f in enumerate(FIRING_FRACTIONS):
            v = aurocs.get((W, f), {}).get("non-C1_vs_C1")
            M[i, j] = v if v is not None else np.nan
    im = ax.imshow(M, aspect="auto", vmin=0.5, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(FIRING_FRACTIONS)))
    ax.set_xticklabels([f"{f:.1f}" for f in FIRING_FRACTIONS])
    ax.set_yticks(range(len(WINDOW_SIZES_CHUNKS)))
    ax.set_yticklabels([f"W={W}" for W in WINDOW_SIZES_CHUNKS])
    ax.set_xlabel("Firing fraction of trace (f)")
    ax.set_ylabel("Window size (chunks)")
    ax.set_title("AUROC: non-C1 vs C1 — runtime gate feasibility")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "auroc_grid.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'auroc_grid.png'}")

    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": {"LOW_EXEC_THRESHOLD": LOW_EXEC_THRESHOLD,
                   "ACTION_HORIZON": ACTION_HORIZON,
                   "WINDOW_SIZES_CHUNKS": WINDOW_SIZES_CHUNKS,
                   "FIRING_FRACTIONS": FIRING_FRACTIONS},
        "aurocs": {f"W={W},f={f:.1f}": v for (W, f), v in aurocs.items()},
        "valid_sample_counts": {f"W={W},f={f:.1f}": n for (W, f), n in valid_samples_count.items()},
    }, indent=2))
    print(f"saved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
