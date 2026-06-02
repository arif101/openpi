"""Trace-length confound check.

LIBERO terminates on success but lets failures run to the time horizon (T=520).
So trace_length AUROC=1.000 on failure-vs-success is essentially a tautology.
The question is whether our ε-features carry information AFTER conditioning
on trace_length — specifically, do they still separate C0 from C1 within
trace-length-matched subgroups?

Three tests:
  (1) Trace-length distribution per stratum — how big is the confound?
  (2) Within FAILURES ONLY (no PHYS_OK), can frac_low_exec_ratio still
      separate C0 vs C1? This eliminates the success-vs-failure confound.
  (3) Within trace-length quartiles, can we still recover the cluster
      structure?
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
OUT_DIR = ANALYSIS_DIR / "phase15_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}
    label_of = {r["path"]: r["label"] for r in summary["rows"]}

    rows = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        T = int(d["action"].shape[0])
        if p.name.startswith("PHYS_OK_"):
            stratum = "PHYS_OK"
        elif p.name in cluster_of:
            stratum = f"C{cluster_of[p.name]}"
        else:
            stratum = "UNK"
        # Pull the ε features from summary
        feat = next((r for r in summary["rows"] if r["path"] == p.name), None)
        if feat is None:
            continue
        rows.append({
            "trace": p.name, "T": T, "stratum": stratum,
            "frac_low_exec_ratio": feat["frac_low_exec_ratio"],
            "max_consecutive_low_exec": feat["max_consecutive_low_exec"],
            "corr_y": feat["corr_y"],
            "persistent_dir_after_high_eps": feat["persistent_dir_after_high_eps"],
        })

    print("=" * 70)
    print("(1) Trace-length distribution per stratum")
    print("=" * 70)
    by = defaultdict(list)
    for r in rows:
        by[r["stratum"]].append(r["T"])
    print(f"{'stratum':10s} {'N':>4s} {'min':>6s} {'p25':>6s} {'median':>7s} {'p75':>6s} {'max':>6s}")
    for s in sorted(by):
        v = np.array(by[s])
        print(f"  {s:8s} {len(v):>4d} {v.min():>6.0f} {np.percentile(v,25):>6.0f} "
              f"{np.median(v):>7.0f} {np.percentile(v,75):>6.0f} {v.max():>6.0f}")

    # Overlap between strata in trace_length
    print("\n=== Trace-length AUROC per contrast (the confound size) ===")
    contrasts = [("PHYS_OK", "C0"), ("PHYS_OK", "C1"), ("C0", "C1")]
    for neg, pos in contrasts:
        neg_vals = [r["T"] for r in rows if r["stratum"] == neg]
        pos_vals = [r["T"] for r in rows if r["stratum"] == pos]
        if len(neg_vals) < 2 or len(pos_vals) < 2:
            continue
        y = np.array([0] * len(neg_vals) + [1] * len(pos_vals))
        s = np.array(neg_vals + pos_vals)
        auc = roc_auc_score(y, s)
        if auc < 0.5:
            auc = 1 - auc
        print(f"  trace_length on {neg:8s} vs {pos:8s}: AUROC = {auc:.3f}  "
              f"(neg N={len(neg_vals)}, pos N={len(pos_vals)})")

    print("\n" + "=" * 70)
    print("(2) Within FAILURES ONLY: does frac_low_exec_ratio still separate C0 from C1?")
    print("=" * 70)
    print("This test removes the failure-vs-success confound entirely.\n")
    failure_rows = [r for r in rows if r["stratum"] in ("C0", "C1")]
    c0_rows = [r for r in failure_rows if r["stratum"] == "C0"]
    c1_rows = [r for r in failure_rows if r["stratum"] == "C1"]
    print(f"  N_C0 = {len(c0_rows)}  N_C1 = {len(c1_rows)}")

    for feat in ["frac_low_exec_ratio", "max_consecutive_low_exec", "corr_y",
                 "persistent_dir_after_high_eps", "T"]:
        c0_v = np.array([r[feat] for r in c0_rows])
        c1_v = np.array([r[feat] for r in c1_rows])
        y = np.array([0] * len(c0_v) + [1] * len(c1_v))
        s = np.concatenate([c0_v, c1_v])
        auc = roc_auc_score(y, s)
        if auc < 0.5:
            auc_use, sign = 1 - auc, "-"
        else:
            auc_use, sign = auc, "+"
        print(f"  {feat:35s}  AUROC C0 vs C1 = {auc_use:.3f}  sign={sign}  "
              f"(C0 mean={c0_v.mean():.3f}, C1 mean={c1_v.mean():.3f})")

    print("\n" + "=" * 70)
    print("(3) Within trace-length quartiles: does cluster structure survive?")
    print("=" * 70)
    Ts = np.array([r["T"] for r in failure_rows])
    quartile_edges = np.percentile(Ts, [25, 50, 75])
    print(f"  Quartile edges (T): p25={quartile_edges[0]:.0f}  median={quartile_edges[1]:.0f}  p75={quartile_edges[2]:.0f}")

    # ε-features within each quartile
    quartiles = {}
    for r in failure_rows:
        T = r["T"]
        if T <= quartile_edges[0]:
            q = "Q1 (shortest)"
        elif T <= quartile_edges[1]:
            q = "Q2"
        elif T <= quartile_edges[2]:
            q = "Q3"
        else:
            q = "Q4 (longest)"
        quartiles.setdefault(q, []).append(r)

    for q in ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]:
        qrows = quartiles.get(q, [])
        if len(qrows) < 4:
            print(f"  {q}: N={len(qrows)} (skipping, too few)")
            continue
        c0_q = [r for r in qrows if r["stratum"] == "C0"]
        c1_q = [r for r in qrows if r["stratum"] == "C1"]
        print(f"  {q}: N_total={len(qrows)}  N_C0={len(c0_q)}  N_C1={len(c1_q)}  "
              f"T_range=[{min(r['T'] for r in qrows)},{max(r['T'] for r in qrows)}]")
        if len(c0_q) < 2 or len(c1_q) < 2:
            print(f"    insufficient for within-quartile AUROC")
            continue
        for feat in ["frac_low_exec_ratio", "max_consecutive_low_exec", "corr_y"]:
            c0_v = np.array([r[feat] for r in c0_q])
            c1_v = np.array([r[feat] for r in c1_q])
            y = np.array([0] * len(c0_v) + [1] * len(c1_v))
            s = np.concatenate([c0_v, c1_v])
            auc = roc_auc_score(y, s)
            if auc < 0.5:
                auc = 1 - auc
            print(f"    {feat:35s}  within-quartile AUROC = {auc:.3f}")

    # Visualization: scatterplot of trace_length vs frac_low_exec_ratio, colored by cluster
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"PHYS_OK": "green", "C0": "tab:blue", "C1": "tab:orange"}
    for s in ["PHYS_OK", "C0", "C1"]:
        v = [r for r in rows if r["stratum"] == s]
        if not v:
            continue
        ax.scatter([r["T"] for r in v], [r["frac_low_exec_ratio"] for r in v],
                   c=colors[s], label=f"{s} (N={len(v)})", alpha=0.7, s=60, edgecolor="black")
    ax.set_xlabel("Trace length T")
    ax.set_ylabel("frac_low_exec_ratio")
    ax.set_title("Confound visualization: T vs frac_low_exec_ratio by stratum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trace_length_confound.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'trace_length_confound.png'}")


if __name__ == "__main__":
    main()
