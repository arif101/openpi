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


def grasp_rollout(env, init_state, tname, grasp_pos, grasp_quat, dmp, apu, kp_rot=3.0, debug=False):
    env.reset(); obs = env.set_init_state(init_state)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    # initialize the DMP at the ROBOT's actual start (object frame), not the demo's
    E0 = np.asarray(obs["robot0_eef_pos"], np.float32)
    eq0 = np.asarray(obs["robot0_eef_quat"], np.float32)
    Eq0 = np.array([eq0[3], eq0[0], eq0[1], eq0[2]], np.float32)
    O0p = sim.data.body_xpos[bid].astype(np.float32).copy()
    O0q = sim.data.body_xquat[bid].astype(np.float32).copy()
    e_rel0, _ = ee_in_object_frame(E0, Eq0, O0p, O0q)
    st = {"x": e_rel0.copy().astype(np.float64), "v": np.zeros(3), "s": 1.0}
    x0_run = e_rel0.astype(np.float64)
    dphase = 1.0 / PHASE_STEPS
    min_d, closed, secured = 9.9, False, 0
    for step in range(GRASP_MAX_STEPS):
        E_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        E_q = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        O_pos = sim.data.body_xpos[bid].astype(np.float32).copy()
        O_quat = sim.data.body_xquat[bid].astype(np.float32).copy()
        e_rel, _ = ee_in_object_frame(E_pos, E_q, O_pos, O_quat)
        dist = float(np.linalg.norm(e_rel - grasp_pos))
        min_d = min(min_d, dist)
        if secured < 18:                                  # PHASE 1: approach + settle + close fingers
            dmp.step(st, grasp_pos.astype(np.float64), dt=dphase, x0=x0_run)
            desired_world = O_pos + quat_rotate(O_quat, st["x"].astype(np.float32))
            a_pos = np.clip((desired_world - E_pos) / apu, -1, 1)
            grip = 1.0 if dist < GRIP_EPS else -1.0       # close fingers once near
            if dist < 0.012:                              # only count tight, settled contact
                closed = True; secured += 1
            else:
                secured = 0
        else:                                             # PHASE 2: lift straight up, gripper closed
            a_pos = np.array([0.0, 0.0, 1.0])
            grip = 1.0
        tgt_q = quat_mul(O_quat, grasp_quat)
        err = quat_mul(tgt_q, quat_conj(quat_normalize(E_q)))
        a_rot = np.clip(kp_rot * axisangle_from_quat(err), -1, 1)
        obs, _, done, _ = env.step([*a_pos, *a_rot, grip])
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            return (True, min_d, closed) if debug else True
    return (False, min_d, closed) if debug else False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True)
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--debug", action="store_true")
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
        r = grasp_rollout(env, init, tname, grasp_pos, grasp_quat, dmp, apu, debug=args.debug)
        if args.debug:
            ok, min_d, closed = r
            print(f"  [{i+1}/{len(test_files)}] {'LIFT' if ok else '----'} "
                  f"min_approach={min_d:.3f}m gripper_closed={closed} (dmp {head+ok}/{i+1})", flush=True)
        else:
            ok = r
            print(f"  [{i+1}/{len(test_files)}] {'LIFT' if ok else '----'} (dmp {head+ok}/{i+1})", flush=True)
        head += ok
    n = len(test_files)
    pert = args.pert_dir.split("pert")[-1]
    print(f"\nDMP-GRASP task {args.task_idx} pert={pert}: DMP-lift {head}/{n}={head/n*100:.1f}%  "
          f"vs  Pi0.5-lift {pi05}/{n}={pi05/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
