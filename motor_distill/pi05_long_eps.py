"""eps-Recovery STEP 1: LIBERO-LONG (libero_10) rollouts of pi0.5 + per-chunk COMMANDED-vs-REALIZED epsilon.
Produces the corpus + the action-grounded detection signal for the head-to-head (eps vs SAFE/Pre-VLA/HELM).
eps = the gap between what the policy COMMANDED (action EE-delta) and what the body REALIZED (actual EE-delta);
the 'execution-stuck / effect-absent' signature = commanded large, realized ~0 -> exactly the class that
vision/progress monitors miss. Logs per-step (commanded3, realized3, gripper, eps) + task success.

Run: PYTHONPATH=third_party/libero:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_long_eps.py --seed 0 --out data/long_eps
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="third_party/libero/libero/libero/bddl_files/libero_10")
    p.add_argument("--out", default="data/long_eps"); p.add_argument("--horizon", type=int, default=520)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    print(f"{len(bddls)} LIBERO-LONG tasks, seed {args.seed}", flush=True)
    succ = []
    for bf in bddls:
        task = pathlib.Path(bf).stem
        instr = task.split("_SCENE")[-1]; instr = re.sub(r"^\d+_", "", instr).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed); env.reset(); obs = env.reset()
        COMM, REAL, GRIP, EPS = [], [], [], []
        chunk = None; ci = 0
        for step in range(args.horizon):
            if chunk is None or ci >= args.replan:
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
            a = chunk[ci].astype(np.float32); ci += 1
            ee0 = np.asarray(obs["robot0_eef_pos"], np.float32)
            obs, _, done, _ = env.step(a[:7].tolist())
            ee1 = np.asarray(obs["robot0_eef_pos"], np.float32)
            comm = a[:3]; real = ee1 - ee0
            COMM.append(comm); REAL.append(real); GRIP.append(float(a[6]))
            EPS.append(float(np.linalg.norm(comm - real)))       # commanded-vs-realized residual
            if done: break
        try: ok = bool(env.env._check_success())
        except Exception: ok = False
        succ.append(int(ok)); env.close()
        eps = np.array(EPS)
        np.savez_compressed(out / f"long_{task[:28]}_s{args.seed}.npz",
                            commanded=np.array(COMM), realized=np.array(REAL), grip=np.array(GRIP),
                            eps=eps, success=int(ok), task=task)
        print(f"  {task[:42]:44s} T={len(EPS):3d} eps_mean={eps.mean():.3f} eps_max={eps.max():.3f} {'OK' if ok else 'FAIL'}", flush=True)
    print(f"\n=== pi0.5 LIBERO-LONG (seed={args.seed}) success: {np.mean(succ):.2f} ({sum(succ)}/{len(succ)}) ===", flush=True)
    print("LONG_EPS_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
