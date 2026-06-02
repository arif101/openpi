"""Exp 1 trainer: learned action heads conditioned on PRIVILEGED physical state.

Trains two heads on pert0 grasp-approach demos, both conditioned on the EE pose
in the target OBJECT frame (object-relative = position-invariant by construction)
toward the grasp goal:
  - NDP   : forcing-net(cond) -> differentiable goal-attractor -> approach trajectory.
  - Plain : net(cond, goal) -> approach trajectory directly (no attractor).
Both output the approach trajectory in object frame [T,3]. Closed-loop eval
(eval_heads.py) re-queries each replan from the live object pose -> re-targets to
a moved (OOD) object for free.

Privileged physical state isolates the HEAD question (does physical conditioning
+ structure cure the appearance->action lookup table?) from the perception
question (world encoder = Exp 2). This is the upper-bound / control.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import rekey
from rekey import ee_in_object_frame, quat_xyzw_to_wxyz, quat_normalize, select_target_object
from ndp import NDP, PlainHead

torch.set_num_threads(min(8, os.cpu_count() or 8))
T = 24


def _resample(traj, T):
    idx = np.linspace(0, len(traj) - 1, T).round().astype(int)
    return traj[idx]


def load_approaches(files, task):
    """Return (x0[N,3], traj[N,T,3], grasp_pos[3]) in object frame from pert0 successes."""
    x0s, trajs, grasp = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        act, ee_p, ee_q = d["action"], d["ee_pos"], d["ee_quat"]
        op, oq, names = d["object_pos"], d["object_quat"], list(d["object_names"])
        ti = select_target_object(op, names)
        clos = np.where(act[:, 6] > 0)[0]
        if len(clos) < 1 or clos[0] < 5:
            continue
        tg = int(clos[0])
        traj = np.stack([ee_in_object_frame(ee_p[t], quat_xyzw_to_wxyz(ee_q[t]), op[t, ti], oq[t, ti])[0]
                         for t in range(tg + 1)])           # approach in object frame
        x0s.append(traj[0]); trajs.append(_resample(traj, T)); grasp.append(traj[-1])
    x0 = np.stack(x0s).astype(np.float32)
    traj = np.stack(trajs).astype(np.float32)
    grasp_pos = np.mean(grasp, 0).astype(np.float32)
    return torch.tensor(x0), torch.tensor(traj), torch.tensor(grasp_pos)


def train_one(kind, x0, traj, grasp_pos, steps, lr=1e-3):
    g = grasp_pos[None].repeat(x0.shape[0], 1)
    if kind == "ndp":
        net = NDP(cond_dim=3, n_dim=3, n_bf=20, T=T)
    else:
        net = PlainHead(cond_dim=3, n_dim=3, T=T)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = net(x0, g, x0)                                # cond=x0, goal=grasp, x0=x0
        loss = ((pred - traj) ** 2).mean()
        loss.backward(); opt.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            print(f"  [{kind}] step {step} loss {loss.item():.5f}", flush=True)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--trace-dir", default="data/keystone/pert0")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--out", default="data/keystone/heads")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n", type=int, default=40)
    args = p.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(args.trace_dir) /
                   f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
    x0, traj, grasp_pos = load_approaches(files, args.task_idx)
    print(f"task {args.task_idx}: {x0.shape[0]} approaches, T={T}, grasp_pos(obj)={grasp_pos.numpy().round(3)}",
          flush=True)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for kind in ("ndp", "plain"):
        net = train_one(kind, x0, traj, grasp_pos, args.steps)
        torch.save({"state_dict": net.state_dict(), "kind": kind, "T": T,
                    "grasp_pos": grasp_pos.numpy()}, out / f"head_{kind}_t{args.task_idx}.pt")
        print(f"  saved {out}/head_{kind}_t{args.task_idx}.pt", flush=True)


if __name__ == "__main__":
    main()
