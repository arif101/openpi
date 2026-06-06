"""DECISIVE DIAGNOSTIC: what does pi0.5's OWN motor achieve WITH CORRECT BINDING?
Run pi0.5 end-to-end (no noise, no our-motor) on the object it binds correctly = the memorized object
(distractors[0], same as collect_dart). Instruction names that object -> pi0.5 binds right. Score with the
OFFICIAL LIBERO BDDL success (env._check_success). This is the CEILING of "good motor + correct binding":
  - HIGH (~40%+) => binding is the lever; pi0.5-motor + correct binding = SOTA path (hybrid).
  - ~baseline (~15-20%) => full pick-place is just hard even for pi0.5's motor; no binding fix reaches SOTA.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_ceiling.py --seed 7
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
    p.add_argument("--container", default="basket"); p.add_argument("--n-scenes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    succ = []
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        if not distractors:
            continue
        M = distractors[0]                                    # pi0.5's correctly-bound (memorized) object
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, [M, args.container + "_1"]); cb = rb[args.container + "_1"]
        except Exception:
            env.close(); continue
        instr = f"pick up the {nm(M)} and place it in the {args.container}"
        z0 = body_pos(sim, rb[M])[2]; lifted = 0.0; chunk = None; ci = 0
        for step in range(args.horizon):
            if chunk is None or ci >= args.replan:
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
            obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
            lifted = max(lifted, body_pos(sim, rb[M])[2] - z0)
            if done:
                break
        # CORRECT check: did pi0.5 place its bound object M IN the basket? Geometric (xy<6cm ~= official containment
        # we validated: OFFICIAL=Y cases all had obj2basket_xy<=5cm). _check_success() wants T (counterfactual), not M.
        mp = body_pos(sim, rb[M]); cp = body_pos(sim, cb)
        d_xy = float(np.linalg.norm(mp[:2] - cp[:2]))
        ok = bool(lifted > 0.04 and d_xy < 0.06)
        succ.append(int(ok)); env.close()
        print(f"  {nm(M):14s} lifted={lifted*100:3.0f}cm M2basket={d_xy*100:4.0f}cm IN_BASKET={'Y' if ok else '.'}", flush=True)
    print(f"\n=== pi0.5 OWN MOTOR + CORRECT BINDING (N={len(succ)}, seed={args.seed}) M-in-basket: {np.mean(succ):.2f} ===", flush=True)
    print("PI05_CEILING_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
