"""Stage 1 KILL/KEEP gate: does the distilled goal-conditioned head REDIRECT by goal?

Closed-loop deploy the head on the captured object scenes with ORACLE goal = the COUNTERFACTUAL
target T's position (the object pi0.5 does NOT go to). If the head reaches T instead of the memorized
M, the motor half is validated (object-agnostic, redirect-by-goal) -- BEFORE we build the binding
channel. Baseline pi0.5 grounding on these scenes is ~20%.

Pure motor test: the head sees only goal-relative proprio, never appearance -> cannot be captured by M.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--head", default="runs/reach_head.pkl")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--horizon", type=int, default=140)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    with open(args.head, "rb") as fh:
        params = jax.tree.map(jnp.asarray, pickle.load(fh))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        rT, rM = 1e9, 1e9
        for step in range(args.horizon):
            Tp = body_pos(sim, rb[T]).astype(np.float32)         # ORACLE goal = counterfactual target
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            ee_rel = jnp.asarray(ee - Tp)
            quat = jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32))
            grip = jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))
            a = np.asarray(head_apply(params, ee_rel, quat, grip))
            obs, _, done, info = env.step(a.tolist())
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            rT = min(rT, float(np.linalg.norm(E - body_pos(sim, rb[T]))))
            rM = min(rM, float(np.linalg.norm(E - body_pos(sim, rb[M]))))
            if done:
                break
        env.close()
        reached = rT < rM
        rows.append(reached)
        print(f"  {pathlib.Path(bf).stem[:26]:28s} T={T:18s} | reach_T={rT*100:.1f}cm reach_M={rM*100:.1f}cm "
              f"reached_T={reached}", flush=True)
    if rows:
        print(f"\n=== ORACLE-GOAL REDIRECT (N={len(rows)}) ===", flush=True)
        print(f"  head reaches counterfactual target: {sum(rows)}/{len(rows)} = {sum(rows)/len(rows)*100:.0f}%  "
              f"(baseline pi0.5 ~20%)", flush=True)
    print("EVAL_REACH_REDIRECT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
