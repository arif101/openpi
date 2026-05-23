"""Diagnose the IN_FALSE failure traces.

Mirror of the IN_TRUE_CLOSE_FALSE diagnostic, but applied to the bowl
trajectory + grasp state instead of the EE trajectory + execution state.

For each IN_FALSE_CLOSE_FALSE (bowl not in drawer, drawer not closed)
and IN_FALSE_CLOSE_TRUE (drawer closed somehow but bowl elsewhere) trace,
compute:

  ee_to_bowl_min : closest approach EE-to-bowl over the trajectory (m)
                   (proxy for "did EE try to grasp the bowl at all")
  bowl_motion   : how much the bowl moved from init position (m)
                   (proxy for "did pickup happen")
  bowl_max_y     : how far in +y the bowl ever got toward the drawer (m)
  bowl_final_y   : where the bowl ended up
  grip_close_count : how many steps the gripper was commanded closed
                     near the bowl (proxy for grasp attempts)

Sub-mode classification:
  NO_APPROACH:  ee_to_bowl_min > 0.08  (EE never reached the bowl)
  GRASP_FAIL:   ee_to_bowl_min < 0.05  AND bowl_motion < 0.02
                (EE near bowl, bowl didn't move — grasp attempt failed)
  TRANSPORT_FAIL: bowl_motion > 0.05  AND bowl_max_y > 0.10  AND
                  bowl_final_y < 0.15  (picked up, then dropped before drawer)
  PLACEMENT_MISS: bowl reached drawer area (max_y > 0.15) but ended outside
                  the drawer site AABB
  OTHER:        anything that doesn't match

This tells us whether IN_FALSE failures cluster into clean sub-modes
(targetable by dense reward) or are scattered (different intervention
needed).
"""

import json
import pathlib
import sys

import numpy as np


def classify(d, drawer_y_goal=0.30, ee_to_bowl_grasp=0.05, ee_to_bowl_approach=0.08):
    ee = d["ee_pos"]            # (T, 3)
    op = d["object_pos"]         # (T, n_obj, 3)
    bowl = op[:, 0]              # (T, 3) — primary tracked object
    act = d["action"]            # (T, 7)
    T = act.shape[0]

    ee_to_bowl = np.linalg.norm(ee - bowl, axis=-1)
    ee_to_bowl_min = float(ee_to_bowl.min())
    ee_to_bowl_min_step = int(np.argmin(ee_to_bowl))

    bowl_init = bowl[0]
    bowl_motion = float(np.linalg.norm(bowl - bowl_init, axis=-1).max())
    bowl_max_y = float(bowl[:, 1].max())
    bowl_max_y_step = int(np.argmax(bowl[:, 1]))
    bowl_final_y = float(bowl[-1, 1])

    # Gripper command analysis (action[6]: +1 close, -1 open in robosuite OSC)
    # Steps where gripper was commanded closed near the bowl
    near_bowl = ee_to_bowl < 0.08
    grip_closed = act[:, 6] > 0.5
    grip_close_near_bowl = int((near_bowl & grip_closed).sum())

    # Sub-mode classification
    if ee_to_bowl_min > ee_to_bowl_approach:
        submode = "NO_APPROACH"
    elif ee_to_bowl_min < ee_to_bowl_grasp and bowl_motion < 0.02:
        submode = "GRASP_FAIL"
    elif bowl_motion > 0.05 and bowl_max_y > 0.10 and bowl_final_y < 0.15:
        submode = "TRANSPORT_FAIL"
    elif bowl_max_y >= 0.15:
        submode = "PLACEMENT_MISS"
    else:
        submode = "OTHER"

    return {
        "submode": submode,
        "ee_to_bowl_min_cm": ee_to_bowl_min * 100,
        "ee_to_bowl_min_step": ee_to_bowl_min_step,
        "bowl_motion_cm": bowl_motion * 100,
        "bowl_max_y_cm": bowl_max_y * 100,
        "bowl_max_y_step": bowl_max_y_step,
        "bowl_final_y_cm": bowl_final_y * 100,
        "grip_close_near_bowl_steps": grip_close_near_bowl,
        "T": T,
    }


def main():
    strata_p = pathlib.Path("/tmp/openpi-data/failure_strata_bddl_v2.json")
    traces_dir = pathlib.Path("/tmp/openpi-data/recovery_source_traces")
    strata = json.loads(strata_p.read_text())

    in_false_names = [
        (n, r) for n, r in strata.items()
        if r.get("label", "").startswith("IN_FALSE")
    ]
    print(f"N IN_FALSE traces: {len(in_false_names)}")
    print(f"  IN_FALSE_CLOSE_FALSE: {sum(1 for _, r in in_false_names if r['label'] == 'IN_FALSE_CLOSE_FALSE')}")
    print(f"  IN_FALSE_CLOSE_TRUE:  {sum(1 for _, r in in_false_names if r['label'] == 'IN_FALSE_CLOSE_TRUE')}")
    print()

    header = f"{'trace':<60} {'BDDL':<22} {'submode':<18} {'ee2bowl':>8} {'bowl_dq':>8} {'bowl_y_max':>10} {'bowl_y_final':>12} {'grip_n':>7}"
    print(header)
    print("-" * len(header))

    results = []
    submode_counts = {}
    for name, r in in_false_names:
        try:
            d = np.load(traces_dir / name, allow_pickle=True)
            c = classify(d)
            c["trace"] = name
            c["bddl_label"] = r["label"]
            results.append(c)
            submode_counts[c["submode"]] = submode_counts.get(c["submode"], 0) + 1
            print(f"{name[:58]:<60} {r['label']:<22} {c['submode']:<18} "
                  f"{c['ee_to_bowl_min_cm']:>6.1f}cm {c['bowl_motion_cm']:>6.1f}cm "
                  f"{c['bowl_max_y_cm']:>+8.1f}cm {c['bowl_final_y_cm']:>+10.1f}cm "
                  f"{c['grip_close_near_bowl_steps']:>7d}")
        except Exception as e:
            print(f"  ERR {name}: {type(e).__name__}: {e}")

    print()
    print(f"Sub-mode distribution across {len(results)} IN_FALSE traces:")
    for sm, count in sorted(submode_counts.items(), key=lambda x: -x[1]):
        print(f"  {sm:<20} {count:>3d}  ({count*100//len(results)}%)")

    # Save
    pathlib.Path("/tmp/openpi-data/in_false_diagnostic.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nSaved /tmp/openpi-data/in_false_diagnostic.json")


if __name__ == "__main__":
    main()
