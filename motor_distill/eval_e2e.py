"""Stage 2/3 END-TO-END eval (NO oracle): binding head infers the goal from the COUNTERFACTUAL
instruction; reach head executes it closed-loop. Does the full factored policy reach the
language-named target T (not memorized M)?

baseline pi0.5 ~20%   oracle-goal reach head 90% (Stage 1)   -> this measures how much the binding
channel recovers of that 90% headroom.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs
from distill_reach import head_apply
from distill_bind import bind_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--bind-head", default="runs/bind_head.pkl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=20, help="re-estimate goal every N steps")
    args = ap.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.models import model as _model
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    with open(args.reach_head, "rb") as fh:
        rp = jax.tree.map(jnp.asarray, pickle.load(fh))
    with open(args.bind_head, "rb") as fh:
        bd = pickle.load(fh); bp = jax.tree.map(jnp.asarray, bd["params"]); n_img = bd["n_img"]

    def infer_goal(img, wr, ee, eq, gq, instr):
        inp = policy._input_transform(build_obs(img, wr, ee, eq, gq, instr))
        inp = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
        o = _model.Observation.from_dict(inp)
        pf, _ = policy._model.extract_vlm_spatial_features(o)
        return np.asarray(bind_apply(bp, pf, n_img)[0])

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)        # instr names T (counterfactual)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                goal = infer_goal(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                  obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                if gerr is None:
                    gerr = float(np.linalg.norm(goal - body_pos(sim, rb[T])))   # binding error vs true T
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
        reached = rT < rM
        rows.append(reached)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={T:16s} | bind_err={gerr*100:.1f}cm "
              f"reach_T={rT*100:.1f}cm reach_M={rM*100:.1f}cm reached_T={reached}", flush=True)
    if rows:
        print(f"\n=== END-TO-END (no oracle, N={len(rows)}) ===", flush=True)
        print(f"  factored policy reaches named target: {sum(rows)}/{len(rows)} = {sum(rows)/len(rows)*100:.0f}%"
              f"   (baseline 20%, oracle-goal 90%)", flush=True)
    print("EVAL_E2E_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
