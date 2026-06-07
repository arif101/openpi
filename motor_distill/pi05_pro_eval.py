"""FULL LIBERO-PRO eval for pi0.5 — per suite x axis, WITH init-state handling (the position/swap axes need the
.pruned_init applied). Reproduces the published baseline (~23.69% overall) = our measuring stick before we build
the capability that beats VLS's 36.81%. Official BDDL success via env._check_success.

Run: PYTHONPATH=third_party/libero:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_pro_eval.py \
       --data-root /root/LIBERO-Pro-data --axes lan,task,object,swap --suites libero_object,libero_goal,libero_spatial,libero_10 --trials 5
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, build_obs

HORIZON = {"libero_10": 520, "libero_goal": 300, "libero_spatial": 220, "libero_object": 280}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--data-root", default="/root/LIBERO-Pro-data")
    p.add_argument("--suites", default="libero_object,libero_goal,libero_spatial,libero_10")
    p.add_argument("--axes", default="lan,task,object,swap")
    p.add_argument("--trials", type=int, default=5)         # init-states per task
    p.add_argument("--replan", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    root = pathlib.Path(args.data_root)
    suites = args.suites.split(","); axes = args.axes.split(",")
    grand = []
    for suite in suites:
        H = HORIZON.get(suite, 300)
        for axis in axes:
            bdir = root / "bddl_files" / f"{suite}_{axis}"
            idir = root / "init_files" / f"{suite}_{axis}"
            bddls = sorted(glob.glob(str(bdir / "*.bddl")))
            if not bddls:
                continue
            succ = []
            for bf in bddls:
                stem = pathlib.Path(bf).stem
                inits = None
                fi = idir / f"{stem}.pruned_init"
                if fi.exists():
                    try: inits = np.load(fi, allow_pickle=True)
                    except Exception: inits = None
                instr = parse_bddl(bf)[0]
                nt = min(args.trials, len(inits)) if inits is not None else args.trials
                for t in range(nt):
                    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
                    env.seed(args.seed + t); env.reset()
                    obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                    chunk = None; ci = 0
                    for step in range(H):
                        if chunk is None or ci >= args.replan:
                            o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                             obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                            chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                        obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                        if done: break
                    try: ok = bool(env.env._check_success())
                    except Exception: ok = False
                    succ.append(int(ok)); env.close()
            grand += succ
            print(f"  {suite}_{axis:7s}: {np.mean(succ):.3f}  ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== pi0.5 LIBERO-PRO OVERALL (axes={args.axes}, trials={args.trials}, seed={args.seed}): "
          f"{np.mean(grand):.4f} ({sum(grand)}/{len(grand)}) ===  [VLS bar 36.81%, pi0.5 paper 23.69%]", flush=True)
    print("PRO_EVAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
