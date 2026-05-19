"""Extract the canonical "drawer push" joint configuration from PHYS_OK traces.

For each successful task-3 trace, find the moment the drawer was actively
closing (qpos[bottom_level] heading toward zero, EE near the drawer face),
and record the robot joint angles at that moment. Mean across traces gives
a canonical push pose for the recovery primitive.

Writes a single 7-element float array to the output path.

Usage:
  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\
      scripts/find_drawer_push_pose.py \\
      --traces-dir data/contact_mpc/recovery_source_traces \\
      --out data/contact_mpc/drawer_push_pose.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

_orig = torch.load
def _patched(*args, **kwargs):
    if "weights_only" not in kwargs: kwargs["weights_only"] = False
    return _orig(*args, **kwargs)
torch.load = _patched

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-suite", default="libero_10")
    ap.add_argument("--task-idx", type=int, default=3)
    args = ap.parse_args()

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env.seed(0)
    env.reset()

    mdl = env.env.sim.model
    drawer_qpos_addr = int(mdl.jnt_qposadr[mdl.joint_name2id("white_cabinet_1_bottom_level")])
    print(f"white_cabinet_1_bottom_level qpos address: {drawer_qpos_addr}")

    traces_dir = pathlib.Path(args.traces_dir)
    ok_paths = sorted(traces_dir.glob(f"PHYS_OK_libero_10_task{args.task_idx}_*.npz"))
    print(f"Inspecting {len(ok_paths)} PHYS_OK traces for push-pose moments.")

    push_poses = []
    push_ees = []
    push_drawer_qpos = []
    for p in ok_paths:
        d = np.load(p, allow_pickle=True)
        qpos = d["qpos"]
        ee = d["ee_pos"]
        T = qpos.shape[0]
        dq = qpos[:, drawer_qpos_addr]

        # Identify drawer closing event: qpos went from < -0.04 (clearly open)
        # to > -0.005 (essentially closed) within the trajectory.
        if dq.min() > -0.04 or dq[-1] > -0.005:
            # Either drawer never opened much or didn't close — skip.
            pass
        # Find first step where dq crosses -0.04 going UP (toward zero / closing).
        push_window = None
        for t in range(1, T):
            if dq[t - 1] < -0.04 and dq[t] > -0.04 and dq[min(t + 20, T - 1)] > -0.02:
                push_window = (max(0, t - 5), min(T, t + 5))
                break
        if push_window is None:
            continue
        a, b = push_window
        # Pose at middle of window
        mid = (a + b) // 2
        push_poses.append(qpos[mid, :7])
        push_ees.append(ee[mid])
        push_drawer_qpos.append(dq[mid])

    env.close()

    if not push_poses:
        print("No clear drawer-close events found.")
        return 1

    push_poses = np.array(push_poses)
    push_ees = np.array(push_ees)
    pdq = np.array(push_drawer_qpos)
    print(f"\nFound push events in {len(push_poses)} traces.\n")
    print(f"Mean drawer qpos at push: {pdq.mean():+.4f}  (closing range)")
    print(f"Mean EE position at push: ({push_ees[:,0].mean():+.3f}, {push_ees[:,1].mean():+.3f}, {push_ees[:,2].mean():+.3f})")
    print(f"EE position std:          ({push_ees[:,0].std():.3f}, {push_ees[:,1].std():.3f}, {push_ees[:,2].std():.3f})")
    print()
    print("Joint angle stats across push events:")
    for j in range(7):
        print(f"  joint{j}: mean={push_poses[:,j].mean():+.4f}  std={push_poses[:,j].std():.4f}")

    canonical = np.median(push_poses, axis=0)
    print(f"\nCanonical push pose (median):\n  {canonical}")

    np.savez(args.out,
             push_pose_qpos=canonical,
             all_push_poses=push_poses,
             all_push_ees=push_ees,
             drawer_qpos_addr=drawer_qpos_addr)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
