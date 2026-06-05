"""END-TO-END with open-vocab detector binding: GroundingDINO(target name) -> box -> box_head -> 3D
goal -> distilled reach head closed-loop. No oracle. Tests the full pipeline's ceiling vs the
pi0.5-feature binding (which inherited capture). baseline 20%, oracle-goal 90%.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, clean_query
from distill_reach import head_apply
from distill_box import apply as box_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--box-head", default="runs/box_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=20)
    args = ap.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach_head, "rb")))
    bp = jax.tree.map(jnp.asarray, pickle.load(open(args.box_head, "rb")))
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
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                box = detect_box(np.asarray(obs["agentview_image"]), clean_query(T), dev)
                if box is not None:
                    goal = np.asarray(box_apply(bp, jnp.asarray(np.array(box, np.float32) / 256.0)))
                    if gerr is None:
                        gerr = float(np.linalg.norm(goal - body_pos(sim, rb[T])))
            if goal is None:
                break
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            a = np.asarray(head_apply(rp, jnp.asarray(ee - goal),
                                      jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                      jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
            obs, _, done, info = env.step(a.tolist())
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            rT = min(rT, float(np.linalg.norm(E - body_pos(sim, rb[T]))))
            rM = min(rM, float(np.linalg.norm(E - body_pos(sim, rb[M]))))
            if done:
                break
        env.close()
        reached = (rT < rM)
        rows.append(reached)
        ge = f"{gerr*100:.1f}cm" if gerr is not None else "n/a"
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={T:16s} | bind_err={ge} "
              f"reach_T={rT*100:.1f}cm reach_M={rM*100:.1f}cm reached_T={reached}", flush=True)
    if rows:
        print(f"\n=== END-TO-END GDINO (no oracle, N={len(rows)}) ===", flush=True)
        print(f"  reaches named target: {sum(rows)}/{len(rows)} = {sum(rows)/len(rows)*100:.0f}%"
              f"   (baseline 20%, pi0.5-feat binding 50%, oracle 90%)", flush=True)
    print("EVAL_E2E_GDINO_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
