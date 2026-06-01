"""A0 — the keystone go/no-go.

Does the distilled goal-relative head reproduce Pi0.5's success when handed
Pi0.5's OWN target plan (the replayed g-sequence)? This isolates the HEAD: no
target_gen, no Pi0.5 at runtime, no pixels. If the head reproduces the task
given correct targets, Pi0.5's motor manifold transferred through the
goal-relative re-keyed distillation (keystone holds). If not, the keystone is
in trouble regardless of any planner.

For each reference trace (a Pi0.5 SUCCESS), we restore its exact init via the
recorded init_state_libero, then roll the head closed-loop (receding horizon),
feeding g_t replayed from that trace's realized targets, and score BDDL done.
Metric = fraction of Pi0.5 successes the head reproduces.

Run on the box (needs the LIBERO sim), CPU-forced for the tiny head:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MUJOCO_GL=egl \
  PYTHONPATH=motor_distill:third_party/libero \
  uv run --no-sync python motor_distill/eval_a0.py \
    --ckpt data/keystone/ckpt/head_equiv.pt --task-idx 3 \
    --trace-dir data/keystone/pert0 --n 20
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import head as H
import rekey

torch.set_num_threads(min(8, os.cpu_count() or 8))

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
NUM_STEPS_WAIT = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}


def load_head(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = H.MotorHead(horizon=ck["horizon"], hidden=ck.get("hidden", 256))
    net.load_state_dict(ck["state_dict"])
    net.equivariant = ck["equivariant"]
    net.eval()
    return net, ck["horizon"]


def _t(x):
    return torch.tensor(np.asarray(x, dtype=np.float32)[None])      # [1, d]


def rollout(net, env, sim, ref, target_bid, replan, max_steps, sample_steps=10):
    """One head-driven rollout with g replayed from the reference trace.
    Returns True iff BDDL success (env done)."""
    env.reset()
    obs = env.set_init_state(ref["init_state_libero"])
    g_pos, g_quat = ref["g_pos"], ref["g_quat"]
    ng = len(g_pos)
    done = False
    t = 0
    # settle (match harness): dummy actions before control
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            return True
    step = 0
    while step < max_steps:
        gi = min(step, ng - 1)                                       # replay g, clamp at end
        ee_pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        eq = np.asarray(obs["robot0_eef_quat"], np.float32)          # xyzw
        ee_quat_wxyz = np.array([eq[3], eq[0], eq[1], eq[2]], np.float32)
        grip = np.asarray(obs["robot0_gripper_qpos"], np.float32)
        proprio = np.concatenate([ee_pos, ee_quat_wxyz, grip])
        obj_pos = sim.data.body_xpos[target_bid].astype(np.float32).copy()
        obj_quat = sim.data.body_xquat[target_bid].astype(np.float32).copy()  # wxyz
        with torch.no_grad():
            chunk = net.sample(_t(g_pos[gi]), _t(g_quat[gi]), _t(proprio),
                               _t(obj_pos), _t(obj_quat), steps=sample_steps)[0].numpy()
        for j in range(min(replan, chunk.shape[0])):
            obs, _, done, _ = env.step(chunk[j].tolist())
            step += 1
            if done:
                return True
            if step >= max_steps:
                break
    return bool(done)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--n", type=int, default=20, help="num reference episodes")
    p.add_argument("--replan", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--sample-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    net, H_ = load_head(args.ckpt)
    print(f"head: {args.ckpt} (equivariant={net.equivariant}, horizon={H_})", flush=True)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(args.seed)
    sim = env.env.sim
    max_steps = MAX_STEPS[args.task_suite]
    print(f"task {args.task_idx}: {task.language}", flush=True)

    pat = f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz"
    files = sorted(glob.glob(str(pathlib.Path(args.trace_dir) / pat)))[: args.n]
    if not files:
        raise FileNotFoundError(f"no traces matching {pat} in {args.trace_dir}")

    succ = 0
    for i, f in enumerate(files):
        out = rekey.build_pairs(f, rekey.RekeyConfig(horizon=args.horizon))
        d = np.load(f, allow_pickle=True)
        target_name = out["target_name"]
        target_bid = int(sim.model.body_name2id(target_name))
        ref = {"init_state_libero": d["init_state_libero"],
               "g_pos": out["g_pos"], "g_quat": out["g_quat"]}
        ok = rollout(net, env, sim, ref, target_bid, args.replan, max_steps, args.sample_steps)
        succ += ok
        print(f"  [{i+1}/{len(files)}] target={target_name:28s} -> {'OK' if ok else 'FAIL'} "
              f"(running {succ}/{i+1})", flush=True)
    print(f"\nA0 result: head reproduces {succ}/{len(files)} = {succ/len(files)*100:.1f}% "
          f"of Pi0.5 successes (task {args.task_idx}, {'equiv' if net.equivariant else 'plain'})", flush=True)


if __name__ == "__main__":
    main()
