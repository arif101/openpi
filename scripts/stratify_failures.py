"""Stratify failure traces by unsatisfied BDDL sub-predicate.

A pooled recovery rate across failure types with different perception/
repertoire answers is uninterpretable. Before any planner test, we need
to know which sub-predicate(s) each failure leaves unsatisfied so we can
report recovery rate per stratum, not in aggregate.

For LIBERO-10 the typical BDDL is a conjunction of sub-predicates. Each
sub-predicate fails in characteristic ways visible in the trajectory:

  task 3: (And (Close drawer) (In bowl drawer))
    - GRIP_LOSS:        bowl never moved >5cm from init position
                        (pickup never stuck)
    - PLACEMENT_FAIL:   bowl moved >5cm but final ee-bowl distance >10cm
                        (grasped & lifted but lost grip)
    - PLACEMENT_OK:     bowl near goal_xyz but drawer not closed
                        (sub-predicate failure: drawer close)
    - APPROACH_FAIL:    EE never got within 10cm of bowl
                        (Pi0.5 disoriented from start)

  task 7: (And (In soup basket) (In cream_cheese basket))
    - GRIP_LOSS_OBJ1, GRIP_LOSS_OBJ2: similar logic per object
    - PARTIAL_PLACEMENT: one object placed, other not

The stratifier produces a JSON mapping trace_filename → label, which the
P3 planner then reads to filter to a single stratum.

Usage:
  PYTHONPATH=src uv run python3 scripts/stratify_failures.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --out data/contact_mpc/failure_strata.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np


def stratify_one(d, goal_xyz: np.ndarray) -> dict:
    """Classify a single failed-trace NPZ by failure mode."""
    obj_pos = d["object_pos"]      # [T, n_obj, 3]
    ee_pos = d["ee_pos"]            # [T, 3]
    if obj_pos.shape[1] == 0:
        return {"label": "UNKNOWN_NO_OBJECTS"}

    # Primary tracked object is index 0 (the task's main object)
    obj0 = obj_pos[:, 0]
    obj0_motion = float(np.linalg.norm(obj0.max(axis=0) - obj0.min(axis=0)))
    obj0_final = obj0[-1]
    ee_to_obj0 = np.linalg.norm(ee_pos - obj0, axis=-1)
    obj0_to_goal = float(np.linalg.norm(obj0_final - goal_xyz))
    ee_obj0_final = float(np.linalg.norm(ee_pos[-1] - obj0_final))
    ee_obj0_min = float(ee_to_obj0.min())

    # APPROACH_FAIL: EE never got close to the object at all
    if ee_obj0_min > 0.10:
        return {
            "label": "APPROACH_FAIL",
            "ee_obj_min_cm": ee_obj0_min * 100,
            "obj_motion_cm": obj0_motion * 100,
            "obj_to_goal_cm": obj0_to_goal * 100,
        }

    # GRIP_LOSS: EE got close, but the object barely moved
    if ee_obj0_min < 0.08 and obj0_motion < 0.05:
        return {
            "label": "GRIP_LOSS",
            "ee_obj_min_cm": ee_obj0_min * 100,
            "obj_motion_cm": obj0_motion * 100,
            "obj_to_goal_cm": obj0_to_goal * 100,
        }

    # PLACEMENT_OK_SUBPRED_FAIL: object made it near the goal but the
    # task is still unsatisfied. For task 3 this is the drawer-close
    # case; for task 7 this is the second-object case if one is placed.
    if obj0_to_goal < 0.10:
        return {
            "label": "PLACEMENT_OK_SUBPRED_FAIL",
            "obj_to_goal_cm": obj0_to_goal * 100,
            "obj_motion_cm": obj0_motion * 100,
        }

    # PLACEMENT_FAIL: object moved meaningfully but ended up far from
    # the goal AND not near the EE — likely dropped mid-transport.
    if obj0_motion > 0.05 and ee_obj0_final > 0.10 and obj0_to_goal > 0.10:
        return {
            "label": "PLACEMENT_FAIL",
            "obj_motion_cm": obj0_motion * 100,
            "ee_obj_final_cm": ee_obj0_final * 100,
            "obj_to_goal_cm": obj0_to_goal * 100,
        }

    return {
        "label": "OTHER",
        "ee_obj_min_cm": ee_obj0_min * 100,
        "obj_motion_cm": obj0_motion * 100,
        "obj_to_goal_cm": obj0_to_goal * 100,
        "ee_obj_final_cm": ee_obj0_final * 100,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    ap.add_argument("--task-suite", default="libero_10")
    args = ap.parse_args()

    import yaml
    targets = yaml.safe_load(pathlib.Path(args.targets_yaml).read_text())

    traces_dir = pathlib.Path(args.traces_dir)
    paths = sorted(traces_dir.glob("PHYS_FAIL_*.npz"))
    print(f"Stratifying {len(paths)} failure traces...")

    out: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for p in paths:
        m = re.search(r"task(\d+)", p.name)
        if not m: continue
        tidx = int(m.group(1))
        entry = targets.get(args.task_suite, {}).get(tidx)
        if not entry: continue
        goal_xyz = np.array(entry["goal_xyz"], dtype=np.float64)
        d = np.load(p, allow_pickle=True)
        info = stratify_one(d, goal_xyz)
        info["task_idx"] = tidx
        info["task_description"] = str(d.get("task_description", ""))
        info["mode"] = str(d.get("mode", ""))
        out[p.name] = info
        counts[info["label"]] = counts.get(info["label"], 0) + 1

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nLabel distribution across {len(out)} traces:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:>3d}  ({v*100//max(len(out),1)}%)")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
