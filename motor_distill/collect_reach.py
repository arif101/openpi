"""Stage 1 data: collect goal-conditioned reaching demos from frozen pi0.5.

For each object scene, for each present object o, roll out pi0.5 with instruction "pick up the {o}"
and log per-step (proprio, goal=o's 3D pos, action). KEEP only rollouts where pi0.5 actually
reaches o (min EE->o < keep_cm) -> clean goal-ALIGNED reaching trajectories with diverse goals.

These distill into a goal-conditioned attractor head: the head learns "reach the goal" from pi0.5's
competent motion, NOT "which object" (the goal is given, not chosen). Saved to data/reach/*.npz.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import re

import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--out", default="data/reach")
    p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--k-init", type=int, default=3)
    p.add_argument("--horizon", type=int, default=110)
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--keep-cm", type=float, default=6.0)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    kept = 0; tried = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        present = list(dict.fromkeys((targets or []) + (distractors or [])))   # graspable objects in scene
        if not present:
            continue
        for o in present:
            oname = re.sub(r"_\d+$", "", o).replace("_", " ")
            for k in range(args.k_init):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
                env.seed(100 + k); env.reset(); obs = env.reset()
                sim = env.env.sim; rb = resolve_bodies(sim, [o])
                tried += 1
                P, Q, G, A, GP = [], [], [], [], []
                chunk = None; ci = 0; reach = 1e9
                for step in range(args.horizon):
                    gpos = body_pos(sim, rb[o]).astype(np.float32)
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    if chunk is None or ci >= args.replan:
                        o_in = build_obs(np.asarray(obs["agentview_image"]),
                                         np.asarray(obs["robot0_eye_in_hand_image"]),
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                         obs["robot0_gripper_qpos"], f"pick up the {oname}")
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                    a = chunk[ci]
                    P.append(ee); Q.append(np.asarray(obs["robot0_eef_quat"], np.float32))
                    G.append(np.asarray(obs["robot0_gripper_qpos"], np.float32))
                    GP.append(gpos); A.append(a[:7].astype(np.float32))
                    obs, _, done, info = env.step(a.tolist()); ci += 1
                    reach = min(reach, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - gpos)))
                    if done:
                        break
                env.close()
                if reach < args.keep_cm / 100.0:           # pi0.5 actually reached o -> keep aligned demo
                    np.savez(out / f"{pathlib.Path(bf).stem[:24]}_{o}_{k}.npz",
                             ee=np.array(P), quat=np.array(Q), grip=np.array(G),
                             goal=np.array(GP), action=np.array(A))
                    kept += 1
                print(f"  {pathlib.Path(bf).stem[:24]:26s} o={oname:14s} k={k} reach={reach*100:.1f}cm "
                      f"{'KEEP' if reach < args.keep_cm/100 else 'drop'}", flush=True)
    print(f"\nKEPT {kept}/{tried} aligned reaching demos -> {out}", flush=True)
    print("COLLECT_REACH_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
