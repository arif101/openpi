"""Grasp-only servo test — isolates the precision crux + robustness claim.

Can the head GRASP+LIFT the object when given a LIVE, object-relative grasp
target (bounded lookahead), recomputed each step from the current object pose?
This is the hardest, most precision-critical subgoal, and grasping a DISPLACED
object via an object-relative target is exactly where pixel-Pi0.5 should struggle
and object-relative servoing should shine.

target_gen (grasp-only): grasp pose = mean EE pose at gripper-close in object
frame, derived from demos. Each step, g = bounded step (capped at the head's
training lookahead magnitude) toward that grasp pose, in the object frame.

Metric = object lifted (z rises > LIFT_M). Reported for in-dist (pert0) and
perturbed (pert5/10), vs Pi0.5's own lift-rate computed from its logged traces.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import rekey
from eval_a0 import load_head, _t
from rekey import ee_in_object_frame, quat_xyzw_to_wxyz, quat_normalize, select_target_object

torch.set_num_threads(min(8, os.cpu_count() or 8))
LIFT_M = 0.05
GRASP_MAX_STEPS = 200


def derive_grasp(files, horizon):
    """Grasp pose (object-frame EE at gripper-close) + max_disp, from demos."""
    pos, quat, gmags, tname = [], [], [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        act, ee_p, ee_q = d["action"], d["ee_pos"], d["ee_quat"]
        op, oq, names = d["object_pos"], d["object_quat"], list(d["object_names"])
        ti = select_target_object(op, names)
        tname = str(names[ti])
        clos = np.where(act[:, 6] > 0)[0]                 # first gripper-close command
        if len(clos) == 0:
            continue
        tg = int(clos[0])
        rp, rq = ee_in_object_frame(ee_p[tg], quat_xyzw_to_wxyz(ee_q[tg]), op[tg, ti], oq[tg, ti])
        pos.append(rp); quat.append(rq)
        o = rekey.build_pairs(f, rekey.RekeyConfig(horizon=horizon))
        gmags.append(np.linalg.norm(o["g_pos"], axis=1))
    grasp_pos = np.mean(pos, 0).astype(np.float32)
    grasp_quat = quat_normalize(np.mean(quat, 0)).astype(np.float32)
    max_disp = float(np.percentile(np.concatenate(gmags), 95))
    return grasp_pos, grasp_quat, max_disp, tname


def pi05_lifted(trace, tname):
    d = np.load(trace, allow_pickle=True)
    names = list(d["object_names"])
    ti = names.index(tname) if tname in names else select_target_object(d["object_pos"], names)
    z = d["object_pos"][:, ti, 2]
    return float(z.max() - z[0]) > LIFT_M


def grasp_rollout(net, env, init_state, tname, grasp_pos, grasp_quat, max_disp, replan, ss):
    env.reset()
    obs = env.set_init_state(init_state)
    sim = env.env.sim
    bid = int(sim.model.body_name2id(tname))
    z0 = float(sim.data.body_xpos[bid][2])
    step = 0
    while step < GRASP_MAX_STEPS:
        E_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)
        E_q = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        grip = np.asarray(obs["robot0_gripper_qpos"], np.float32)
        O_pos = sim.data.body_xpos[bid].astype(np.float32).copy()
        O_quat = sim.data.body_xquat[bid].astype(np.float32).copy()
        e_rel, _ = ee_in_object_frame(E_pos, E_q, O_pos, O_quat)       # EE in object frame
        delta = grasp_pos - e_rel                                      # toward grasp pose
        n = np.linalg.norm(delta)
        if n > max_disp:
            delta = delta * (max_disp / n)
        g_pos = (e_rel + delta).astype(np.float32)                     # bounded lookahead
        proprio = np.concatenate([E_pos, E_q, grip])
        with torch.no_grad():
            chunk = net.sample(_t(g_pos), _t(grasp_quat), _t(proprio),
                               _t(O_pos), _t(O_quat), steps=ss)[0].numpy()
        for j in range(min(replan, chunk.shape[0])):
            obs, _, done, _ = env.step(chunk[j].tolist())
            step += 1
            if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
                return True
            if step >= GRASP_MAX_STEPS:
                break
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True, help="scenes to test (pert0/5/10)")
    p.add_argument("--ref-dir", default="data/keystone/pert0", help="demos -> grasp pose")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--replan", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    net, _ = load_head(args.ckpt)
    tag = "equiv" if net.equivariant else "plain"
    ref_files = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                       f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    grasp_pos, grasp_quat, max_disp, tname = derive_grasp(ref_files[:30], args.horizon)
    print(f"task {args.task_idx} [{tag}] grasp_pos(obj)={grasp_pos.round(3)} max_disp={max_disp:.3f} "
          f"target={tname}", flush=True)

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
        ok = grasp_rollout(net, env, init, tname, grasp_pos, grasp_quat, max_disp,
                           args.replan, args.sample_steps)
        head += ok
        print(f"  [{i+1}/{len(test_files)}] {'LIFT' if ok else '----'} (head {head}/{i+1})", flush=True)
    n = len(test_files)
    pert = args.pert_dir.split("pert")[-1]
    print(f"\nGRASP [{tag}] task {args.task_idx} pert={pert}: HEAD-lift {head}/{n}={head/n*100:.1f}%  "
          f"vs  Pi0.5-lift {pi05}/{n}={pi05/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
