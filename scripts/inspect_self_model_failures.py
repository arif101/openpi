"""Inspect the traces where the feasibility self-model fails (R² < 0).

Which traces, which strata, what's anomalous about them?
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")


def main() -> None:
    res = json.loads((ANALYSIS_DIR / "self_model_phase1" / "results.json").read_text())
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())

    # The per_trace_r2 list is in the order of sorted(TRACES_DIR.glob("*.npz")).
    trace_paths = sorted(TRACES_DIR.glob("*.npz"))
    trace_names = [p.name for p in trace_paths]
    label_of = {r["path"]: r["label"] for r in summary["rows"]}

    print(f"variants checked: {list(res['results'].keys())}\n")

    for variant in ["commanded", "proprio", "both"]:
        r2s = np.array(res["results"][variant]["per_trace_r2"])
        if len(r2s) != len(trace_names):
            print(f"WARN: r2 count {len(r2s)} != trace count {len(trace_names)}")
        print(f"=== variant={variant}: traces with R² < 0 ===")
        bad_idx = np.where(r2s < 0)[0]
        print(f"  n_bad = {len(bad_idx)}")

        # By stratum: among R²<0 traces, what are the labels?
        from collections import Counter
        bad_labels = Counter(label_of.get(trace_names[i], "UNK") for i in bad_idx)
        all_labels = Counter(label_of.get(n, "UNK") for n in trace_names)
        print(f"  by stratum:")
        for lab in sorted(all_labels):
            n_bad = bad_labels.get(lab, 0)
            n_tot = all_labels.get(lab, 0)
            print(f"    {lab:30s}  {n_bad:>3d} / {n_tot:>3d}  ({100 * n_bad / n_tot:>5.1f}%)")
        print()

    # Detailed: list the specific traces that failed under 'both', with their per-trace true ε
    print("=== detailed list (variant='both', sorted by ascending R²) ===")
    r2_both = np.array(res["results"]["both"]["per_trace_r2"])
    true_mean = np.array(res["results"]["both"]["per_trace_true_mean"])
    pred_mean = np.array(res["results"]["both"]["per_trace_pred_mean"])
    order = np.argsort(r2_both)
    for k, i in enumerate(order[:12]):
        nm = trace_names[i]
        lab = label_of.get(nm, "UNK")
        # Trace length
        d = np.load(trace_paths[i], allow_pickle=True)
        T = d["action"].shape[0]
        print(f"  R²={r2_both[i]:>+7.3f}  T={T:>4d}  true_mean={true_mean[i]:.3f}  pred_mean={pred_mean[i]:.3f}"
              f"  label={lab:25s}  {nm}")

    # Hypothesis check: are R²<0 traces the SHORT ones?
    print("\n=== Hypothesis: are R²<0 traces unusually short or have low ε variance? ===")
    lengths = []
    eps_stds = []
    for p in trace_paths:
        d = np.load(p, allow_pickle=True)
        lengths.append(d["action"].shape[0])
    lengths = np.array(lengths)

    bad = r2_both < 0
    good = r2_both > 0.5
    print(f"R²<0:     N={bad.sum():3d}, mean length={lengths[bad].mean():.0f}, median={np.median(lengths[bad]):.0f}")
    print(f"R²>0.5:   N={good.sum():3d}, mean length={lengths[good].mean():.0f}, median={np.median(lengths[good]):.0f}")

    # Variance of true target per trace?
    # We need to reload summary stats — easier: variance of mean ε across the trace's samples is not in results.json.
    # Use a proxy: the across-trace true_mean variability vs the trace's own true_mean.
    # Skip detailed variance — we'll just note the length pattern.


if __name__ == "__main__":
    main()
