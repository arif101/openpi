"""DMP grasp test — does a re-targetable goal-attractor primitive grasp+lift the
object where the distilled head got 0%, and re-target to perturbed objects by
construction?

Replaces the distilled flow head with a DMP whose goal g = live object-relative
grasp pose. The attractor guarantees convergence to g; a forcing term fit from
ONE demo's approach gives the approach shape. g is recomputed each step from the
LIVE object pose, so it re-aims to a displaced object for free.

Metric = object lifted (z rises > LIFT_M), in-dist (pert0) and perturbed
(pert5/10), vs Pi0.5's own lift-rate. Compare to the distilled head's 0% grasp.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np

import rekey
from rekey import (ee_in_object_frame, quat_xyzw_to_wxyz, quat_normalize,
                   quat_mul, quat_conj, quat_rotate, select_target_object)
from eval_grasp import pi05_lifted, LIFT_M, GRASP_MAX_STEPS
from dmp import DMP

GRIP_EPS = 0.025          # close gripper when EE within this of grasp pose (object frame)
PHASE_STEPS = 130         # env steps over which the DMP phase completes


def axisangle_from_quat(q):
    q = quat_normalize(q)
    w = np.clip(q[0], -1, 1)
    ang = 2 * np.arccos(w)
    sn = np.sqrt(max(1 - w * w, 1e-9))
    axis = q[1:] / sn if sn > 1e-6 else np.zeros(3)
    if ang > np.pi:
        ang -= 2 * np.pi
    return axis * ang


def derive(files, horizon):
    """grasp pose (object frame) + fitted DMP + action scaling, from demos."""
    pos, quat, tname = [], [], None
    dmp_traj = None
    ee_d_all, act_all = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        act, ee_p, ee_q = d["action"], d["ee_pos"], d["ee_quat"]
        op, oq, names = d["object_pos"], d["object_quat"], list(d["object_names"])
        ti = select_target_object(op, names); tname = str(names[ti])
        clos = np.where(act[:, 6] > 0)[0]
        if len(clos) == 0:
            continue
        tg = int(clos[0])
        rp, rq = ee_in_object_frame(ee_p[tg], quat_xyzw_to_wxyz(ee_q[tg]), op[tg, ti], oq[tg, ti])
        pos.append(rp); quat.append(rq)
        # action->displacement scale (meters per action unit)
        eed = np.diff(ee_p[:tg + 1], axis=0); a = act[:tg, :3]
        m = np.abs(a) > 0.1
        if m.any():
            ee_d_all.append(np.abs(eed[m])); act_all.append(np.abs(a[m]))
        # DMP demo: approach EE trajectory in object frame (start -> grasp)
        if dmp_traj is None and tg > 20:
            traj = np.stack([ee_in_object_frame(ee_p[t], quat_xyzw_to_wxyz(ee_q[t]),
                             op[t, ti], oq[t, ti])[0] for t in range(tg + 1)])
            dmp_traj = traj
    grasp_pos = np.mean(pos, 0).astype(np.float32)
    grasp_quat = quat_normalize(np.mean(quat, 0)).astype(np.float32)
    act_per_unit = float(np.median(np.concatenate(ee_d_all) / np.concatenate(act_all)))
    dmp = DMP(n_dim=3, n_bf=25).fit(dmp_traj)
    return grasp_pos, grasp_quat, tname, dmp, act_per_unit


def grasp_rollout(env, init_state, tname, grasp_pos, grasp_quat, dmp, apu, kp_rot=3.0):
    env.reset(); obs = env.set_init_state(init_state)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    st = dmp.reset(); dphase = 1.0 / PHASE_STEPS
    for step in range(GRASP_MAX_STEPS):
        E_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        E_q = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        O_pos = sim.data.body_xpos[bid].astype(np.float32).copy()
        O_quat = sim.data.body_xquat[bid].astype(np.float32).copy()
        # advance DMP toward grasp_pos (object frame); desired pos -> world
        dmp.step(st, grasp_pos, dt=dphase, x0=dmp.x0)
        desired_world = O_pos + quat_rotate(O_quat, st["x"].astype(np.float32))
        a_pos = np.clip((desired_world - E_pos) / apu, -1, 1)
        # orientation: proportional toward grasp orientation (object->world)
        tgt_q = quat_mul(O_quat, grasp_quat)
        err = quat_mul(tgt_q, quat_conj(quat_normalize(E_q)))
        a_rot = np.clip(kp_rot * axisangle_from_quat(err), -1, 1)
        # gripper: close when EE near grasp pose (object frame)
        e_rel, _ = ee_in_object_frame(E_pos, E_q, O_pos, O_quat)
        grip = 1.0 if np.linalg.norm(e_rel - grasp_pos) < GRIP_EPS else -1.0
        obs, _, done, _ = env.step([*a_pos, *a_rot, grip])
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True)
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ref_files = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                       f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    grasp_pos, grasp_quat, tname, dmp, apu = derive(ref_files[:30], args.horizon)
    print(f"task {args.task_idx} [DMP] grasp_pos(obj)={grasp_pos.round(3)} act/unit={apu:.4f} target={tname}",
          flush=True)

    test_files = sorted(glob.glob(str(pathlib.Path(args.pert_dir) /
                        f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
    pi05 = sum(pi05_lifted(f, tname) for f in test_files)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)

    head = 0
    for i, f in enumerate(test_files):
        init = np.load(f, allow_pickle=True)["init_state_libero"]
        ok = grasp_rollout(env, init, tname, grasp_pos, grasp_quat, dmp, apu)
        head += ok
        print(f"  [{i+1}/{len(test_files)}] {'LIFT' if ok else '----'} (dmp {head}/{i+1})", flush=True)
    n = len(test_files)
    pert = args.pert_dir.split("pert")[-1]
    print(f"\nDMP-GRASP task {args.task_idx} pert={pert}: DMP-lift {head}/{n}={head/n*100:.1f}%  "
          f"vs  Pi0.5-lift {pi05}/{n}={pi05/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
