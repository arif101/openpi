"""DECISIVE Option-B viability test (cheap, inference-only): is pi0.5's SWAP failure fixable by VISUAL ISOLATION,
or is the position-memorization ABSOLUTE? De-attractor GRAYING failed (OOD gray patches). Here we HIDE the
distractor objects (set geom rgba alpha=0 -> a clean, in-distribution single-object scene) and ask: does pi0.5 now
grasp the RELOCATED target on the swap axis?
  - If swap lifts >> 17% -> pi0.5 CAN be redirected by isolating the target -> mask-conditioning (Option B) is viable.
  - If swap stays ~17% (goes to empty memorized spot) -> position memorization is ABSOLUTE -> Option B is dead too.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_hide_distractors.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap --n 10 --trials 3
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, build_obs


def geom_ids_for_body(sim, body_name):
    bid = sim.model.body_name2id(body_name)
    return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--res", type=int, default=256)
    p.add_argument("--mode", default="both", choices=["baseline", "hide", "both"])
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    modes = ["baseline", "hide"] if args.mode == "both" else [args.mode]
    for mode in modes:
        succ = []
        for bf in bddls:
            instr, objs, targets, distractors = parse_bddl(bf)
            if not targets: continue
            T = targets[0]; stem = pathlib.Path(bf).stem
            distract_objs = [o for o in objs if args.container not in o and o != T]
            inits = None
            if args.init_dir:
                fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
                if fi.exists():
                    try: inits = np.asarray(torch.load(fi, weights_only=False))
                    except Exception: inits = None
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res)
                env.seed(t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                if mode == "hide":
                    rb = resolve_bodies(sim, distract_objs)
                    for o in distract_objs:
                        if rb.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rb[o]):
                            sim.model.geom_rgba[g, 3] = 0.0          # render-invisible (physics unchanged)
                    obs = env.set_init_state(inits[t]) if inits is not None else obs  # re-render with alpha applied
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
            print(f"  [{mode}] {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
        print(f"\n=== pi0.5 {mode.upper()} (distractors hidden={mode=='hide'}) on {pathlib.Path(args.bddl_dir).name}: "
              f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)}) ===  [swap baseline ~17%]", flush=True)
    print("HIDE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
