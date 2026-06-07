"""DIAGNOSTIC (sizes the capability, NOT a shipped wrapper): does giving pi0.5 the object's TRUE current 3D
position fix the SWAP axis (where it collapses to ~28% via position memorization)? Steer the frozen policy
toward the privileged true object position (then container after grasp) via sample_actions_guided. If swap
lifts sharply -> position perception IS the lever -> build it in (foveation, trained). If not -> rethink.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_swap_priv.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap --guide-w 0.1
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--guide-w", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import jax, jax.numpy as jnp, torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download, nnx_utils
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    GUIDE = {"dir": np.zeros(3, np.float32), "w": float(args.guide_w)}
    gsamp = nnx_utils.module_jit(policy._model.sample_actions_guided)
    def guided(rng, obs, **kw):
        kw.pop("guide_dir", None); kw.pop("guide_w", None)
        return gsamp(rng, obs, guide_dir=jnp.asarray(GUIDE["dir"])[None, :], guide_w=GUIDE["w"], **kw)
    policy._sample_actions = guided
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        nt = min(args.trials, len(inits)) if inits is not None else args.trials
        for t in range(nt):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(args.seed + t); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
            try:
                rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
            except Exception:
                env.close(); succ.append(0); continue
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                active = body_pos(sim, cb) if held else body_pos(sim, rb[T])   # PRIVILEGED true positions
                d = active.astype(np.float32) - ee; nmd = np.linalg.norm(d)
                GUIDE["dir"] = (d / nmd).astype(np.float32) if nmd > 1e-6 else np.zeros(3, np.float32)
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                obs, _, done, _ = env.step(a[:7].tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
        print(f"  {nm(T):16s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== SWAP-axis PRIVILEGED-position guided pi0.5 (guide_w={args.guide_w}, N={len(succ)}): {np.mean(succ):.3f} "
          f"({sum(succ)}/{len(succ)}) ===  [baseline swap ~0.28; if HIGH -> position is the lever]", flush=True)
    print("SWAP_PRIV_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
