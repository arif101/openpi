"""Streaming-mode simulation of the gate, to validate offline-vs-online consistency.

The concern: in offline analysis we evaluate the gate at fixed firing fractions
(f=0.1, 0.2, ...). In deployment, the gate runs at every step t and fires as
soon as the windowed feature crosses the threshold. The two questions:

  (A) Does the strict-online version (1-step lookahead for each step in window)
      give the same numbers as a hindsight version (5-step lookahead, "wait
      for OSC controller to respond")? If they diverge meaningfully, our
      offline measurement biases the gate.

  (B) When we simulate the gate at every step, at what step does it first
      fire for each stratum? Are the firing times informative?
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "phase2a_chunk_level"

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01
WINDOW_CHUNKS = 3            # W=3 chunks = 30 steps
WINDOW_STEPS = WINDOW_CHUNKS * ACTION_HORIZON
GATE_THRESHOLD = 0.133       # locked from Youden analysis


def compute_streaming_features(path: pathlib.Path, lookahead: int = 1) -> dict:
    """Compute the windowed gate signal at every step t, using only data
    available by step t. With `lookahead=1`, realized[s] = ee_pos[s+1] - ee_pos[s].
    With `lookahead=k`, realized[s] = ee_pos[s+k] - ee_pos[s].
    A larger lookahead waits for the OSC controller to converge before judging
    "did the EE move." But it's only computable up to step T-k.
    """
    d = np.load(path, allow_pickle=True)
    action = d["action"]                     # (T, 7)
    ee_pos = d["ee_pos"]                     # (T, 3)
    T = action.shape[0]

    # commanded
    cmd_norm = np.linalg.norm(action[:T - lookahead, :3], axis=1) + 1e-9

    # realized with k-step lookahead
    realized = ee_pos[lookahead:lookahead + (T - lookahead)] - ee_pos[:T - lookahead]
    real_norm = np.linalg.norm(realized, axis=1)
    low_exec = (real_norm / cmd_norm < LOW_EXEC_THRESHOLD).astype(np.float32)
    # pad so low_exec has length T (last `lookahead` steps are 0)
    low_exec_padded = np.concatenate([low_exec, np.zeros(lookahead, dtype=np.float32)])

    # Compute windowed feature at every step t
    windowed = np.zeros(T, dtype=np.float32)
    for t in range(WINDOW_STEPS, T):
        windowed[t] = low_exec_padded[t - WINDOW_STEPS:t].mean()
    return {"T": T, "low_exec": low_exec_padded, "windowed": windowed}


def main() -> None:
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    trace_results = []
    for p in sorted(TRACES_DIR.glob("*.npz")):
        if p.name.startswith("PHYS_OK_"):
            stratum = "PHYS_OK"
        elif p.name in cluster_of:
            stratum = f"C{cluster_of[p.name]}"
        else:
            continue
        # Strict-online: 1-step lookahead
        s1 = compute_streaming_features(p, lookahead=1)
        # Hindsight: 5-step lookahead (controller convergence)
        s5 = compute_streaming_features(p, lookahead=5)
        trace_results.append({"name": p.name, "stratum": stratum,
                              "T": s1["T"], "w_online": s1["windowed"], "w_hindsight": s5["windowed"]})

    # (A) Compare strict-online vs hindsight windowed features at every step
    print("(A) Online vs hindsight: do the windowed features agree?")
    all_diffs = []
    for tr in trace_results:
        valid = tr["w_online"][WINDOW_STEPS:tr["T"] - 5]
        valid_hs = tr["w_hindsight"][WINDOW_STEPS:tr["T"] - 5]
        diffs = valid - valid_hs
        all_diffs.append(diffs)
    diffs = np.concatenate(all_diffs)
    print(f"  N samples: {len(diffs)}")
    print(f"  online - hindsight: mean={diffs.mean():+.4f}  median={np.median(diffs):+.4f}  std={diffs.std():.4f}")
    print(f"  p5={np.percentile(diffs, 5):+.3f}  p95={np.percentile(diffs, 95):+.3f}")
    # Are the gate decisions the same? Threshold = 0.133
    gate_online = np.concatenate([tr["w_online"][WINDOW_STEPS:tr["T"]-5] for tr in trace_results]) > GATE_THRESHOLD
    gate_hs = np.concatenate([tr["w_hindsight"][WINDOW_STEPS:tr["T"]-5] for tr in trace_results]) > GATE_THRESHOLD
    print(f"  gate firing agreement: {(gate_online == gate_hs).mean():.3%}")
    print(f"  online fires when hindsight doesn't: {((gate_online == 1) & (gate_hs == 0)).sum()} samples (online over-fires)")
    print(f"  hindsight fires when online doesn't: {((gate_online == 0) & (gate_hs == 1)).sum()} samples")

    # (B) Per-trace earliest gate firing step (strict-online)
    print("\n(B) Earliest gate firing step per trace, by stratum (strict-online):")
    by_stratum = defaultdict(list)
    for tr in trace_results:
        firing_steps = np.where(tr["w_online"] > GATE_THRESHOLD)[0]
        if len(firing_steps) == 0:
            by_stratum[tr["stratum"]].append((None, tr["T"]))
        else:
            by_stratum[tr["stratum"]].append((int(firing_steps[0]), tr["T"]))

    for stratum in ["PHYS_OK", "C0", "C1"]:
        rows = by_stratum.get(stratum, [])
        if not rows:
            continue
        fired = [r for r in rows if r[0] is not None]
        never = [r for r in rows if r[0] is None]
        print(f"\n  {stratum} (N={len(rows)}): fired={len(fired)}, never_fired={len(never)}")
        if fired:
            fire_steps = np.array([r[0] for r in fired])
            Ts = np.array([r[1] for r in fired])
            frac_at_fire = fire_steps / Ts
            print(f"    fire step: median={int(np.median(fire_steps))}  p25={int(np.percentile(fire_steps,25))}  p75={int(np.percentile(fire_steps,75))}")
            print(f"    fraction of trace at fire: median={np.median(frac_at_fire):.2f}  p25={np.percentile(frac_at_fire,25):.2f}  p75={np.percentile(frac_at_fire,75):.2f}")

    # (B') Per-trace earliest gate firing step (hindsight, for comparison)
    print("\n(B') Same but with hindsight (5-step lookahead) — what we'd get if we waited:")
    for tr in trace_results:
        firing_steps = np.where(tr["w_hindsight"] > GATE_THRESHOLD)[0]
        tr["fire_hindsight"] = int(firing_steps[0]) if len(firing_steps) else None

    by_stratum_hs = defaultdict(list)
    for tr in trace_results:
        by_stratum_hs[tr["stratum"]].append((tr["fire_hindsight"], tr["T"]))
    for stratum in ["PHYS_OK", "C0", "C1"]:
        rows = by_stratum_hs.get(stratum, [])
        fired = [r for r in rows if r[0] is not None]
        if fired:
            fire_steps = np.array([r[0] for r in fired])
            print(f"  {stratum}: fired={len(fired)}/{len(rows)}, "
                  f"median fire step (hindsight) = {int(np.median(fire_steps))}")

    # (C) Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: trace examples
    examples = {}
    for stratum in ["PHYS_OK", "C0", "C1"]:
        for tr in trace_results:
            if tr["stratum"] == stratum:
                examples[stratum] = tr
                break
    colors = {"PHYS_OK": "tab:green", "C0": "tab:blue", "C1": "tab:orange"}
    for stratum, tr in examples.items():
        T = tr["T"]
        steps = np.arange(T)
        axes[0].plot(steps, tr["w_online"], color=colors[stratum], label=f"{stratum} (online)", linewidth=2)
        axes[0].plot(steps, tr["w_hindsight"], color=colors[stratum], linestyle="--", alpha=0.5, label=f"{stratum} (hindsight)")
    axes[0].axhline(GATE_THRESHOLD, color="red", linestyle=":", label=f"gate threshold = {GATE_THRESHOLD}")
    axes[0].set_xlabel("Step t")
    axes[0].set_ylabel("Windowed frac_low_exec (last 30 steps)")
    axes[0].set_title("Streaming gate signal: one example per stratum")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: histogram of earliest firing step by stratum
    for stratum in ["PHYS_OK", "C0", "C1"]:
        rows = by_stratum.get(stratum, [])
        fired_steps = [r[0] for r in rows if r[0] is not None]
        if fired_steps:
            axes[1].hist(fired_steps, bins=15, alpha=0.6, color=colors[stratum],
                         label=f"{stratum} (N={len(fired_steps)}/{len(rows)})", edgecolor="black")
    axes[1].set_xlabel("Earliest gate firing step")
    axes[1].set_ylabel("# traces")
    axes[1].set_title("When does the gate first fire? (strict-online)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "streaming_simulation.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'streaming_simulation.png'}")


if __name__ == "__main__":
    main()
