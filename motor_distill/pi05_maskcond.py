"""OPTION B decisive test (inference-only): mask-condition pi0.5 via a VISUAL PROMPT instead of action-steering.
On the SWAP axis (pi0.5 collapses to ~17% from POSITION memorization), use instance-seg (idealized binder) to
DE-ATTRACTOR the scene -- gray out the distractor objects so the named target is the salient object pi0.5 must act
on. If pi0.5's swap success jumps, visual-prompt conditioning of pi0.5's OWN precise/goal-correcting motor works
(the core of Option B: external general binding -> visual prompt -> precise motor; no memorization, no fragile
steering, no <2cm-goal requirement). Compares: baseline pi0.5 vs de-attractor-masked pi0.5 on the swap axis.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_maskcond.py \
       --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap \
       --n 10 --trials 3 --mode both
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def seg_id_at(sim, body, cam, R, seg):
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    r = int(min(max(px[0], 0), R - 1)); c = int(min(max(px[1], 0), R - 1))
    return int(seg.reshape(R, R)[r, c])


def deattractor(img_native, seg_native, distractor_ids, val=128):
    """Gray the distractor-object pixels in the (native) agentview image."""
    seg = seg_native.reshape(img_native.shape[0], img_native.shape[1])
    out = img_native.copy()
    for did in distractor_ids:
        if did and did != 0: out[seg == did] = val
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--res", type=int, default=256)
    p.add_argument("--mode", default="both", choices=["baseline", "mask", "both"])
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    R = args.res; AGV = "agentview"
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    modes = ["baseline", "mask"] if args.mode == "both" else [args.mode]
    for mode in modes:
        succ = []
        for bf in bddls:
            instr, objs, targets, distractors = parse_bddl(bf)
            if not targets: continue
            T = targets[0]; stem = pathlib.Path(bf).stem
            graspables = [o for o in objs if args.container not in o and o != T]   # distractor objects to gray
            inits = None
            if args.init_dir:
                fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
                if fi.exists():
                    try: inits = np.asarray(torch.load(fi, weights_only=False))
                    except Exception: inits = None
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R,
                                         camera_segmentations="instance" if mode == "mask" else None)
                env.seed(t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, graspables) if mode == "mask" else {}
                chunk = None; ci = 0
                for step in range(args.horizon):
                    if chunk is None or ci >= args.replan:
                        agv = np.asarray(obs[AGV + "_image"])
                        if mode == "mask":
                            seg = np.asarray(obs[AGV + "_segmentation_instance"])
                            dids = [seg_id_at(sim, rb[o], AGV, R, seg) for o in graspables if rb.get(o) is not None]
                            agv = deattractor(agv, seg, set(dids))
                        o_in = build_obs(agv, np.asarray(obs["robot0_eye_in_hand_image"]),
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                    obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                succ.append(int(ok)); env.close()
            print(f"  [{mode}] {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
        print(f"\n=== pi0.5 {mode.upper()} on {pathlib.Path(args.bddl_dir).name}: {np.mean(succ):.3f} ({sum(succ)}/{len(succ)}) ===  [swap baseline ~17%]", flush=True)
    print("MASKCOND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
