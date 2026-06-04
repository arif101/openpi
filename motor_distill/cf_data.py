"""Counterfactual training-data extractor for Method A (target-binding adapter).

For each LIBERO-PRO TASK scene (same scene the policy memorized; instruction names a
SWAPPED target), dump K init-state observations + the privileged signals needed to
train + supervise the binding adapter:
  obs:  agentview(256), wrist(256), ee_pos/quat/gripper  (pi0.5 inputs)
  lang: the counterfactual instruction (from bddl :language)
  target: named-object position (reach supervision) + GDINO box (grounding signal)
  distractors: memorized-object positions + boxes (for vision-independence / de-attractor)
Saved per (scene, seed) to data/cf_train/*.npz. No pi0.5 inference here — pure data.

Supervision plan (v1): directional reach loss — the bound policy's net EE displacement
should point toward the NAMED target (and away from the memorized distractor). The
named-target position is the privileged signal; we never use it at inference.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, clean_query


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_10_task")
    p.add_argument("--out", default="data/cf_train")
    p.add_argument("--n-scenes", type=int, default=40)
    p.add_argument("--k-init", type=int, default=4)
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    print(f"{len(bddls)} scenes x {args.k_init} inits", flush=True)
    saved = 0
    for bf in bddls:
        instruction, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        scene = pathlib.Path(bf).stem[:34]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        for k in range(args.k_init):
            env.seed(100 + k); env.reset(); obs = env.reset()
            sim = env.env.sim
            rb = resolve_bodies(sim, targets + distractors)
            img = np.asarray(obs["agentview_image"])
            tgt_box = detect_box(img, clean_query(targets[0]), dev)
            dis_box = detect_box(img, clean_query(distractors[0]), dev)
            rec = {
                "agentview": img.astype(np.uint8),
                "wrist": np.asarray(obs["robot0_eye_in_hand_image"]).astype(np.uint8),
                "ee_pos": np.asarray(obs["robot0_eef_pos"], np.float32),
                "ee_quat": np.asarray(obs["robot0_eef_quat"], np.float32),
                "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float32),
                "instruction": instruction,
                "target_name": targets[0],
                "target_pos": body_pos(sim, rb[targets[0]]).astype(np.float32),
                "target_box": np.array(tgt_box if tgt_box is not None else [-1, -1, -1, -1], np.int32),
                "distractor_name": distractors[0],
                "distractor_pos": body_pos(sim, rb[distractors[0]]).astype(np.float32),
                "distractor_box": np.array(dis_box if dis_box is not None else [-1, -1, -1, -1], np.int32),
            }
            np.savez(out / f"{scene}_k{k}.npz", **rec); saved += 1
        env.close()
        print(f"  {scene}: tgt={targets[0]} distr={distractors[0]} ({args.k_init} inits)", flush=True)
    print(f"\nSAVED {saved} training observations to {out}", flush=True)
    print("CF_DATA_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
