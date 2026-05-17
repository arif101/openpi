"""Analyze a failure-trace NPZ produced by run_reason_v3_mppi --log-failure-traces.

Answers the question that determines our capability fix: when MPPI couldn't
rescue a stuck Pi0.5, was the EE going to the wrong place (intent failure → fix
via Phase B distillation) or near the right place but missing (control failure
→ fix via guided exploration in MPPI)?

Usage:
    python3 scripts/analyze_failure_trace.py path/to/FAIL_*.npz
    python3 scripts/analyze_failure_trace.py --dir data/contact_mpc/failure_traces_libero10

Outputs a textual diagnosis plus optional matplotlib plots (--plot). The
diagnosis is decidable from the prints alone — plots are gravy.

Plain stdlib + numpy. Matplotlib only if --plot.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def analyze(path: pathlib.Path, plot: bool = False) -> dict:
    """Load one failure trace and emit a textual diagnosis. Returns a dict of
    key metrics for further aggregation."""
    d = np.load(path, allow_pickle=True)
    ee = d["ee_pos"]                       # [T, 3]
    gripper = d["gripper_cmd"]             # [T]
    ee_delta = d["ee_action_delta"]        # [T, 6] EE delta actions commanded
    steps = int(d["steps_total"])
    obj_keys = [k for k in d.files if k.startswith("obj_") and k.endswith("_xyz")]

    print(f"\n{'='*70}")
    print(f"File: {path.name}")
    print(f"  mode={str(d['mode'])}  task='{str(d['task_description'])}'")
    print(f"  steps={steps}  refinements={int(d['refinements_run'])}")
    print(f"{'='*70}")

    # 1. EE motion — is the policy commanding any movement at all?
    ee_disp = np.linalg.norm(ee[-1] - ee[0])
    ee_path_len = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    ee_bbox = ee.max(axis=0) - ee.min(axis=0)
    print(f"\nEE motion")
    print(f"  start xyz:           {np.round(ee[0], 3)}")
    print(f"  final xyz:           {np.round(ee[-1], 3)}")
    print(f"  net displacement:    {ee_disp:.3f} m")
    print(f"  total path length:   {ee_path_len:.3f} m")
    print(f"  bounding box (xyz):  {np.round(ee_bbox, 3)}")

    # 2. EE-vs-objects — did it ever get near any tracked object?
    print(f"\nEE vs tracked objects")
    closest_per_object = {}
    for k in obj_keys:
        obj = d[k]                                 # [T, 3]
        per_step = np.linalg.norm(ee - obj, axis=1)
        idx = int(np.argmin(per_step))
        closest_per_object[k] = (float(per_step.min()), idx, float(per_step.mean()))
        print(f"  {k}: closest={per_step.min():.3f}m at t={idx}, "
              f"mean={per_step.mean():.3f}m, "
              f"object displaced={np.linalg.norm(obj[-1]-obj[0]):.3f}m")

    # 3. Gripper activity — did the policy try to grasp?
    gripper_changes = int(np.sum(np.abs(np.diff(np.sign(gripper))) > 0))
    print(f"\nGripper activity")
    print(f"  range:        {gripper.min():+.2f} to {gripper.max():+.2f}")
    print(f"  open→close transitions: {gripper_changes}")

    # 4. Commanded delta magnitudes — is policy commanding meaningful motion?
    delta_norms = np.linalg.norm(ee_delta, axis=1)
    print(f"\nCommanded EE deltas")
    print(f"  mean |delta|: {delta_norms.mean():.4f}")
    print(f"  max  |delta|: {delta_norms.max():.4f}")
    print(f"  fraction near-zero (<0.01): {float(np.mean(delta_norms < 0.01)):.2f}")

    # 5. Diagnosis verdict
    print(f"\nDiagnosis")
    closest_overall = min(c[0] for c in closest_per_object.values())
    if ee_disp < 0.03 and delta_norms.mean() < 0.005:
        verdict = (
            "STUCK — policy commanding near-zero motion. Likely intent failure. "
            "Capability fix → Phase B distillation."
        )
    elif closest_overall > 0.15:
        verdict = (
            f"WRONG INTENT — EE never got within 15cm of any tracked object "
            f"(closest {closest_overall:.3f}m). Pi0.5 is reaching to wrong xyz. "
            f"Capability fix → Phase B distillation on refined-state rollouts."
        )
    elif closest_overall < 0.08 and gripper_changes == 0:
        verdict = (
            f"RIGHT INTENT / BAD CONTROL — EE got within {closest_overall:.3f}m "
            "but never grasped. Sub-cm precision issue. Capability fix → "
            "guided exploration in MPPI (mix heuristic candidates that close "
            "gripper at peak proximity)."
        )
    elif closest_overall < 0.08 and gripper_changes > 0:
        verdict = (
            f"FAILED MANIPULATION — EE reached object ({closest_overall:.3f}m), "
            "gripper actuated, but task still failed (object not at goal). "
            "Likely lost grip mid-trajectory or pushed object wrong way. "
            "Capability fix → richer Phase B distillation or contact-aware MPPI."
        )
    else:
        verdict = (
            f"INDETERMINATE — closest_to_obj={closest_overall:.3f}m, "
            f"ee_disp={ee_disp:.3f}m, gripper_changes={gripper_changes}. "
            "Inspect camera frames manually."
        )
    print(f"  → {verdict}")

    if plot:
        _plot(d, path)

    return {
        "file": path.name,
        "mode": str(d["mode"]),
        "steps": steps,
        "ee_displacement_m": ee_disp,
        "ee_path_length_m": ee_path_len,
        "closest_obj_distance_m": closest_overall,
        "gripper_changes": gripper_changes,
        "mean_delta_norm": float(delta_norms.mean()),
        "verdict_key": verdict.split(" —")[0],
    }


def _plot(d, path: pathlib.Path) -> None:
    """Optional plot: EE xyz vs each tracked object over time, plus a couple
    camera frames stacked at the bottom."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping")
        return
    ee = d["ee_pos"]
    t = d["t"]
    obj_keys = [k for k in d.files if k.startswith("obj_") and k.endswith("_xyz")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, comp in enumerate("xyz"):
        axes[i].plot(t, ee[:, i], "k-", label="EE", linewidth=1.5)
        for k in obj_keys:
            axes[i].plot(t, d[k][:, i], "--", label=k.replace("obj_", "").replace("_xyz", ""))
        axes[i].set_ylabel(f"{comp} (m)")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(path.name, fontsize=10)
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="One trace NPZ to analyze.")
    ap.add_argument("--dir", help="Or a directory — analyze every FAIL_*.npz under it.")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.dir:
        files = sorted(pathlib.Path(args.dir).glob("FAIL_*.npz"))
        if not files:
            print(f"No FAIL_*.npz in {args.dir}", file=sys.stderr)
            return 1
        rows = [analyze(f, plot=args.plot) for f in files]
        # Compact summary table
        print(f"\n{'='*70}\nSummary across {len(rows)} traces")
        print(f"{'='*70}")
        verdicts: dict[str, int] = {}
        for r in rows:
            v = r["verdict_key"]
            verdicts[v] = verdicts.get(v, 0) + 1
        for v, c in sorted(verdicts.items(), key=lambda x: -x[1]):
            print(f"  {c:3d}  {v}")
        return 0
    elif args.path:
        analyze(pathlib.Path(args.path), plot=args.plot)
        return 0
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
