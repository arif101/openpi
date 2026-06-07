"""KEYSTONE PROBE (frozen pi0.5, NO training): is pi0.5's manipulation precision carried by the WRIST
(eye-in-hand) camera? The whole factored-motor thesis rests on this unproven claim ("add a wrist-cam to an
object-blind goal-conditioned motor => precision without memorization"). Test it directly by ABLATION.

3 arms, run where pi0.5 is ALREADY STRONG (the `lan` axis = paraphrased instruction at MEMORIZED positions =>
binding is a non-issue, ~100% baseline => we isolate pure MOTOR EXECUTION precision):
  - full  : both cameras (control; reproduces ~1.0)
  - wrist : wrist_image ZEROED  (kill eye-in-hand view)
  - base  : image (base/agentview) ZEROED  (kill the global view)
DECISIVE READ:
  - wrist-drop >> base-drop  => precision lives in the WRIST cam => factored wrist-cam motor JUSTIFIED.
  - wrist-drop ~ base-drop   => generic input-shock, not wrist-specific => thesis WEAKENED, rethink.
  - neither drops            => cameras not load-bearing for execution here => thesis WRONG.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_wristcam_ablation.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_lan --init-dir /root/LIBERO-Pro-data/init_files/libero_object_lan \
       --arms full,wrist,base --n 10 --trials 3 --horizon 280
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--arms", default="full,wrist,base")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    arms = args.arms.split(",")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]

    def run_arm(arm):
        succ = []
        for bf in bddls:
            instr = parse_bddl(bf)[0]; stem = pathlib.Path(bf).stem
            inits = None
            if args.init_dir:
                fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
                if fi.exists():
                    try: inits = np.asarray(torch.load(fi, weights_only=False))
                    except Exception: inits = None
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
                env.seed(args.seed + t); env.reset()
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                chunk = None; ci = 0
                for step in range(args.horizon):
                    if chunk is None or ci >= args.replan:
                        base = np.asarray(obs["agentview_image"]); wr = np.asarray(obs["robot0_eye_in_hand_image"])
                        if arm == "wrist": wr = np.zeros_like(wr)
                        if arm == "base":  base = np.zeros_like(base)
                        o_in = build_obs(base, wr, obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                         obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                    obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                succ.append(int(ok)); env.close()
        return np.mean(succ), sum(succ), len(succ)

    res = {}
    for arm in arms:
        m, s, n = run_arm(arm)
        res[arm] = m
        print(f"##### ARM {arm:6s}: {m:.3f} ({s}/{n}) #####", flush=True)
    base_full = res.get("full", None)
    print("\n=== WRIST-CAM ABLATION (frozen pi0.5, lan axis = memorized pos, motor-precision isolated) ===", flush=True)
    for arm in arms:
        drop = (base_full - res[arm]) if (base_full is not None and arm != "full") else 0.0
        print(f"  {arm:6s} = {res[arm]:.3f}   drop_vs_full = {drop:+.3f}", flush=True)
    if "wrist" in res and "base" in res and base_full is not None:
        wd = base_full - res["wrist"]; bd = base_full - res["base"]
        verdict = ("WRIST carries precision -> factored wrist-cam motor JUSTIFIED" if wd > bd + 0.15
                   else "wrist~base input-shock -> thesis WEAKENED, rethink" if abs(wd - bd) <= 0.15
                   else "BASE matters more than wrist -> wrist-cam thesis WRONG")
        print(f"\n  wrist_drop={wd:+.3f}  base_drop={bd:+.3f}  =>  {verdict}", flush=True)
    print("WRISTCAM_ABLATION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
