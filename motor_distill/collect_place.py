"""Collect FULL pick+place demos from frozen pi0.5 (matching instruction) to distill a phase-aware
place head. For each object scene, instruct pi0.5 to pick the MEMORIZED object M (its competent behavior)
and place it in the basket; log per-step (proprio, target=M pos, container=basket pos, action, phase).
Phase A=approach/grasp (M not yet lifted), B=transport/place (M lifted). Keep episodes where M ends in
the basket. -> data/place/*.npz  (ee, quat, grip, tgt, cont, action, phase)
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--out", default="data/place")
    p.add_argument("--container", default="basket")
    p.add_argument("--n-scenes", type=int, default=10); p.add_argument("--k-init", type=int, default=4)
    p.add_argument("--horizon", type=int, default=260); p.add_argument("--replan", type=int, default=5)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    kept = 0; tried = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not distractors:
            continue
        M = distractors[0]                              # pi0.5's memorized/competent target
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(100 + k); env.reset(); obs = env.reset()
            sim = env.env.sim
            try:
                rb = resolve_bodies(sim, [M, args.container + "_1"]); cb = rb[args.container + "_1"]
            except Exception:
                env.close(); continue
            tried += 1
            instr = f"pick up the {nm(M)} and place it in the {args.container}"
            z0 = body_pos(sim, rb[M])[2]
            EE, Q, G, TG, CG, A, PH = [], [], [], [], [], [], []
            chunk = None; ci = 0; lifted = 0.0
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                a = chunk[ci]
                mp = body_pos(sim, rb[M]).astype(np.float32); cp = body_pos(sim, cb).astype(np.float32)
                EE.append(np.asarray(obs["robot0_eef_pos"], np.float32)); Q.append(np.asarray(obs["robot0_eef_quat"], np.float32))
                G.append(np.asarray(obs["robot0_gripper_qpos"], np.float32)); TG.append(mp); CG.append(cp)
                A.append(a[:7].astype(np.float32)); PH.append(1 if lifted > 0.03 else 0)   # 0=pick,1=place
                obs, _, done, info = env.step(a.tolist()); ci += 1
                lifted = max(lifted, body_pos(sim, rb[M])[2] - z0)
                if done:
                    break
            mp = body_pos(sim, rb[M]); cp = body_pos(sim, cb)
            success = bool(lifted > 0.04 and np.linalg.norm(mp[:2] - cp[:2]) < 0.12)
            env.close()
            if success:
                np.savez(out / f"{pathlib.Path(bf).stem[:22]}_{M}_{k}.npz",
                         ee=np.array(EE), quat=np.array(Q), grip=np.array(G), tgt=np.array(TG),
                         cont=np.array(CG), action=np.array(A), phase=np.array(PH))
                kept += 1
            print(f"  {pathlib.Path(bf).stem[:24]:26s} M={nm(M):14s} k={k} lifted={lifted*100:.0f}cm "
                  f"{'PLACED-KEEP' if success else 'drop'}", flush=True)
    print(f"\nKEPT {kept}/{tried} successful pick+place demos -> {out}", flush=True)
    print("COLLECT_PLACE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
