"""Stock pi0.5 on the LIBERO-PRO TASK axis = the FLOOR + harness-validity check.
For each LIBERO-PRO bddl, instruct pi0.5 with the bddl's own (counterfactual) instruction naming the target T,
roll out, score with the OFFICIAL BDDL predicate (env._check_success). Expected ~0-13% (the known TASK-axis
collapse from memorization) -> validates the harness reproduces published behavior before we train anything.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_baseline.py --seed 7
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
    p.add_argument("--n", type=int, default=10); p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)        # instr NAMES the counterfactual target T
        T = targets[0] if targets else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed); env.reset(); obs = env.reset()
        chunk = None; ci = 0
        for step in range(args.horizon):
            if chunk is None or ci >= args.replan:
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
            obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
            if done: break
        try: ok = bool(env.env._check_success())
        except Exception: ok = False
        succ.append(int(ok)); env.close()
        print(f"  {nm(T) if T else '?':16s} instr='{instr[:40]}' OFFICIAL={'Y' if ok else '.'}", flush=True)
    print(f"\n=== STOCK pi0.5 LIBERO-PRO TASK axis (N={len(succ)}, seed={args.seed}) OFFICIAL success: {np.mean(succ):.2f} ===", flush=True)
    print("PI05_BASELINE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
