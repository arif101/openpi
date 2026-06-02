"""Phase 2a.1: streaming decision rule design.

With k=5 + threshold 0.160, naive first-crossing gives TPR=1.0 but FPR=0.58.
PHYS_OK traces have transient threshold crossings. Two fixes to test:

  (1) Sustained-crossing debounce: require windowed_frac > threshold for N
      consecutive chunks.
  (2) Step-restriction: don't allow gate to fire before step t_min.
      LIBERO PHYS_OK traces have median T=222, C1 T=520. Restricting at
      t_min=260 means "fire only after the rollout has been running longer
      than a typical successful one" — runtime-heuristic, not data-leakage.

Sweep both, find the (N, t_min) pair with best TPR at FPR ≤ 0.15.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "phase2a1_decision_rule"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01
LOOKAHEAD_K = 5
WINDOW_CHUNKS = 5
WINDOW_STEPS = WINDOW_CHUNKS * ACTION_HORIZON
GATE_THRESHOLD = 0.160


def compute_streaming_signal(path):
    d = np.load(path, allow_pickle=True)
    action = d["action"]
    ee_pos = d["ee_pos"]
    T = action.shape[0]
    if T <= LOOKAHEAD_K:
        return None
    cmd_xyz = action[:T - LOOKAHEAD_K, :3]
    realized = ee_pos[LOOKAHEAD_K:] - ee_pos[:T - LOOKAHEAD_K]
    cmd_norm = np.linalg.norm(cmd_xyz, axis=1) + 1e-9
    real_norm = np.linalg.norm(realized, axis=1)
    low_exec = (real_norm / cmd_norm < LOW_EXEC_THRESHOLD).astype(np.float32)
    low_exec_padded = np.concatenate([low_exec, np.zeros(LOOKAHEAD_K, dtype=np.float32)])
    windowed = np.zeros(T, dtype=np.float32)
    for t in range(WINDOW_STEPS, T):
        windowed[t] = low_exec_padded[t - WINDOW_STEPS:t].mean()
    return {"T": T, "windowed": windowed}


def first_fire_step(windowed, t_min, n_sustained):
    """Earliest step t where windowed has been > threshold for n_sustained
    consecutive chunks (n_sustained * ACTION_HORIZON steps), and t >= t_min."""
    above = windowed > GATE_THRESHOLD
    required = n_sustained * ACTION_HORIZON
    if required <= 0:
        # baseline first-crossing
        for t in range(max(t_min, WINDOW_STEPS), len(windowed) - LOOKAHEAD_K):
            if above[t]:
                return t
        return None
    # running count of consecutive Trues
    cnt = 0
    for t in range(WINDOW_STEPS, len(windowed) - LOOKAHEAD_K):
        if above[t]:
            cnt += 1
            if cnt >= required and t >= t_min:
                return t
        else:
            cnt = 0
    return None


def main():
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    traces = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        s = compute_streaming_signal(p)
        if s is None:
            continue
        if p.name.startswith("PHYS_OK_"):
            stratum = "PHYS_OK"
        elif p.name in cluster_of:
            stratum = f"C{cluster_of[p.name]}"
        else:
            continue
        traces.append({"name": p.name, "stratum": stratum, **s})
    print(f"loaded {len(traces)} traces (k={LOOKAHEAD_K}, W={WINDOW_CHUNKS}, threshold={GATE_THRESHOLD})")
    print(f"  PHYS_OK median T = {int(np.median([t['T'] for t in traces if t['stratum']=='PHYS_OK']))}")
    print(f"  C0 median T      = {int(np.median([t['T'] for t in traces if t['stratum']=='C0']))}")
    print(f"  C1 median T      = {int(np.median([t['T'] for t in traces if t['stratum']=='C1']))}")

    # Sweep (n_sustained, t_min) grid
    n_sustained_values = [0, 1, 2, 3, 5, 10]   # 0 = first-crossing baseline
    t_min_values = [0, 100, 150, 200, 260, 300]

    print(f"\nSweep: sustained-crossing N (chunks) x minimum-fire-step t_min")
    print(f"Gate fires at first t such that windowed > {GATE_THRESHOLD} for N consecutive chunks AND t >= t_min")
    print()

    results = {}
    for N in n_sustained_values:
        for t_min in t_min_values:
            by_stratum = {"PHYS_OK": [], "C0": [], "C1": []}
            for tr in traces:
                fire = first_fire_step(tr["windowed"], t_min, N)
                by_stratum[tr["stratum"]].append(fire)
            tpr = sum(1 for x in by_stratum["C1"] if x is not None) / len(by_stratum["C1"])
            non_c1 = by_stratum["PHYS_OK"] + by_stratum["C0"]
            fpr = sum(1 for x in non_c1 if x is not None) / len(non_c1)
            # Median fire step among those that fired
            c1_fires = [x for x in by_stratum["C1"] if x is not None]
            median_fire = int(np.median(c1_fires)) if c1_fires else None
            results[(N, t_min)] = {"tpr": tpr, "fpr": fpr,
                                    "n_C1_fired": sum(1 for x in by_stratum["C1"] if x is not None),
                                    "n_PHYS_OK_fired": sum(1 for x in by_stratum["PHYS_OK"] if x is not None),
                                    "n_C0_fired": sum(1 for x in by_stratum["C0"] if x is not None),
                                    "median_C1_fire_step": median_fire}

    # Print TPR table
    print("TPR table (C1 catch rate):")
    header_label = "N\\t_min"
    print(f"  {header_label:>10s}  " + "  ".join(f"t={t:>4d}" for t in t_min_values))
    for N in n_sustained_values:
        row = f"  N={N:>3d}    "
        for t_min in t_min_values:
            r = results[(N, t_min)]
            row += f"   {r['tpr']:.2f}"
        print(row)

    print("\nFPR table (PHYS_OK + C0 false-fire rate):")
    header_label = "N\\t_min"
    print(f"  {header_label:>10s}  " + "  ".join(f"t={t:>4d}" for t in t_min_values))
    for N in n_sustained_values:
        row = f"  N={N:>3d}    "
        for t_min in t_min_values:
            r = results[(N, t_min)]
            row += f"   {r['fpr']:.2f}"
        print(row)

    # Find best operating point: max TPR subject to FPR <= 0.15
    print("\n=== Operating points satisfying FPR ≤ 0.15, ranked by TPR ===")
    candidates = [(N, t_min, results[(N, t_min)]) for N, t_min in results]
    candidates = [c for c in candidates if c[2]["fpr"] <= 0.15]
    candidates.sort(key=lambda c: -c[2]["tpr"])
    print(f"  {'N':>3s}  {'t_min':>5s}  {'TPR':>5s}  {'FPR':>5s}  {'C1_fired':>10s}  {'PHYS_OK_fired':>15s}  {'C0_fired':>10s}  {'median_C1_fire':>15s}")
    for N, t_min, r in candidates[:10]:
        print(f"  {N:>3d}  {t_min:>5d}  {r['tpr']:>5.2f}  {r['fpr']:>5.2f}  "
              f"{r['n_C1_fired']:>10d}  {r['n_PHYS_OK_fired']:>15d}  {r['n_C0_fired']:>10d}  "
              f"{str(r['median_C1_fire_step']):>15s}")

    if candidates:
        best = candidates[0]
        N, t_min, r = best
        print(f"\nBEST: N={N} sustained chunks, t_min={t_min}, TPR={r['tpr']:.2f}, FPR={r['fpr']:.2f}")
        print(f"  Latency contribution: sustained-crossing adds {N * ACTION_HORIZON} steps ({N * ACTION_HORIZON * 50}ms at 20Hz)")
        print(f"  Plus k={LOOKAHEAD_K} hindsight latency: +{LOOKAHEAD_K * 50}ms")
        print(f"  Total decision latency from first stuck moment: {(N * ACTION_HORIZON + LOOKAHEAD_K) * 50}ms")
        print(f"  Plus t_min restriction: gate cannot fire before step {t_min} (= {t_min * 50}ms = {t_min * 50 / 1000:.1f}s)")

    # Visualization: TPR vs FPR scatter for all (N, t_min) combos
    fig, ax = plt.subplots(figsize=(10, 6))
    for (N, t_min), r in results.items():
        marker = ["o", "s", "D", "P", "*", "X"][n_sustained_values.index(N) % 6]
        ax.scatter(r["fpr"], r["tpr"], s=80, alpha=0.7, marker=marker,
                   c=t_min_values.index(t_min), cmap="viridis", vmin=0, vmax=len(t_min_values))
        ax.annotate(f"N={N},t={t_min}", (r["fpr"], r["tpr"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.15, color="red", linestyle="--", alpha=0.5, label="FPR=0.15 limit")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("FPR (non-C1 false-fire rate)")
    ax.set_ylabel("TPR (C1 catch rate)")
    ax.set_title("Streaming decision-rule sweep: sustained-N × t_min restriction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "decision_rule_sweep.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'decision_rule_sweep.png'}")

    (OUT_DIR / "results.json").write_text(json.dumps({
        "config": {"k": LOOKAHEAD_K, "W": WINDOW_CHUNKS, "threshold": GATE_THRESHOLD,
                   "low_exec_cutoff": LOW_EXEC_THRESHOLD},
        "sweep": {f"N={N},t_min={t_min}": r for (N, t_min), r in results.items()},
    }, indent=2))
    print(f"saved {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
