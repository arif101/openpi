"""Exp 1 eval: closed-loop grasp with a LEARNED head (NDP or Plain), 3-arm OOD.

Each step, query the head from the LIVE EE-in-object-frame for an approach
trajectory (object frame), track a short-lookahead waypoint (re-planning every
step = receding horizon), convert to world via the LIVE object pose -> re-targets
to a moved (OOD) object for free. Reuses eval_dmp_grasp's secure+lift + scaling.

Reports head-lift vs Pi0.5-lift (arm A baseline) for pert 0/5/10. With both
head kinds this gives the 3-arm comparison: appearance-Pi0.5 vs Plain vs NDP.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import rekey
from rekey import ee_in_object_frame, quat_normalize, quat_mul, quat_conj, quat_rotate
from eval_dmp_grasp import pi05_lifted, axisangle_from_quat, LIFT_M, GRASP_MAX_STEPS, GRIP_EPS, derive
from ndp import NDP, PlainHead

torch.set_num_threads(min(8, os.cpu_count() or 8))
LOOKAHEAD = 4


def load_head(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    T = ck["T"]
    net = NDP(cond_dim=3, n_dim=3, n_bf=20, T=T) if ck["kind"] == "ndp" else PlainHead(cond_dim=3, n_dim=3, T=T)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck["kind"], np.asarray(ck["grasp_pos"], np.float32)


def rollout(net, grasp_pos, grasp_quat, env, init, tname, apu, kp_rot=3.0):
    env.reset(); obs = env.set_init_state(init)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    gp = torch.tensor(grasp_pos[None]); secured = 0
    for step in range(GRASP_MAX_STEPS):
        E = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        Eq = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        O = sim.data.body_xpos[bid].astype(np.float32).copy()
        Oq = sim.data.body_xquat[bid].astype(np.float32).copy()
        e_rel, _ = ee_in_object_frame(E, Eq, O, Oq)
        dist = float(np.linalg.norm(e_rel - grasp_pos))
        if secured < 18:                                    # approach: head re-plans each step
            with torch.no_grad():
                c = torch.tensor(e_rel[None], dtype=torch.float32)
                traj = net(c, gp, c)[0].numpy()             # [T,3] object frame
            tgt_obj = traj[min(LOOKAHEAD, len(traj) - 1)]
            desired_world = O + quat_rotate(Oq, tgt_obj.astype(np.float32))
            a_pos = np.clip((desired_world - E) / apu, -1, 1)
            grip = 1.0 if dist < GRIP_EPS else -1.0
            secured = secured + 1 if dist < 0.012 else 0
        else:                                               # lift
            a_pos = np.array([0.0, 0.0, 1.0]); grip = 1.0
        tq = quat_mul(Oq, grasp_quat)
        a_rot = np.clip(kp_rot * axisangle_from_quat(quat_mul(tq, quat_conj(quat_normalize(Eq)))), -1, 1)
        obs, _, done, _ = env.step([*a_pos, *a_rot, grip])
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True)
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    net, kind, grasp_pos = load_head(args.ckpt)
    # grasp orientation + tname + action scaling from demos (position grasp_pos comes from the ckpt)
    ref = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                 f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    _, grasp_quat, tname, _, apu = derive(ref[:30], 16)

    test = sorted(glob.glob(str(pathlib.Path(args.pert_dir) /
                  f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
    pi05 = sum(pi05_lifted(f, tname) for f in test)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    head = 0
    for i, f in enumerate(test):
        init = np.load(f, allow_pickle=True)["init_state_libero"]
        ok = rollout(net, grasp_pos, grasp_quat, env, init, tname, apu)
        head += ok
        print(f"  [{i+1}/{len(test)}] {'LIFT' if ok else '----'} ({kind} {head}/{i+1})", flush=True)
    n = len(test); pert = args.pert_dir.split("pert")[-1]
    print(f"\nEXP1 [{kind}] task {args.task_idx} pert={pert}: {kind}-lift {head}/{n}={head/n*100:.1f}%  "
          f"vs Pi0.5-lift {pi05}/{n}={pi05/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
