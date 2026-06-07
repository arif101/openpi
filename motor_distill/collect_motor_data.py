"""STAGE 1 (data) of the factored wrist-cam motor. Roll out frozen pi0.5 on standard libero_object (it succeeds
~100%), and at each inference step log a DISTILLATION SAMPLE for an IDENTITY-AGNOSTIC, POSITION-AGNOSTIC motor:
  INPUT  (student sees only this):
    - wrist_img  : eye-in-hand view (local; the validated source of precision) resized 128x128 uint8
    - goal_rel   : 3D vector (active_target_pos - ee_pos); active = container after grasp, object before.
                   (privileged sim pos here; at deployment the binder supplies it) -> NO absolute position, NO id
    - proprio    : ee orientation axis-angle(3) + gripper_qpos(2)   -> embodiment only, no identity
    - held       : grasp-phase flag (0/1)
  TARGET (distill pi0.5's motor competence):
    - chunk      : pi0.5's full 10x7 delta-EE action chunk produced at that inference step
Keep only SUCCESSFUL rollouts (distill good behavior). One npz per rollout.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/collect_motor_data.py \
       --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object \
       --init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object \
       --out data/motor_demos --trials 6 --seeds 0,1 --container basket
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs, _quat2axisangle


def surface_point(sim, body, depth_norm, R, cam="agentview"):
    """3D point the BINDER returns: object's projected-center pixel -> depth -> unproject (camera-facing surface).
    Training the motor on THIS (instead of body-center) makes motor+binder goal-consistent (absorbs the systematic
    surface-vs-center offset that crashes a center-trained motor)."""
    import robosuite.utils.camera_utils as cu
    pos = body_pos(sim, body)
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
    r = int(min(max(px[0], 0), R - 1)); c = int(min(max(px[1], 0), R - 1))
    real = cu.get_real_depth_map(sim, depth_norm)
    pts = cu.transform_from_pixels_to_world(np.array([[r, c]]), real[None], np.linalg.inv(w2p))
    return np.asarray(pts[0], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--out", default="data/motor_demos"); p.add_argument("--container", default="basket")
    p.add_argument("--trials", type=int, default=6); p.add_argument("--seeds", default="0,1")
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--img", type=int, default=128)
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi_client import image_tools
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    print(f"{len(bddls)} tasks, seeds={seeds}, trials={args.trials}", flush=True)
    n_ok = 0; n_samp = 0
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
        for seed in seeds:
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
                env.seed(seed + t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); continue
                z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0
                obj_sp = surface_point(sim, rb[T], np.asarray(obs["agentview_depth"]), 256); cont_sp = None
                WR, GR, PR, HE, CH = [], [], [], [], []
                chunk = None; ci = 0
                for step in range(args.horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    if held and cont_sp is None: cont_sp = surface_point(sim, cb, np.asarray(obs["agentview_depth"]), 256)
                    active = cont_sp if (held and cont_sp is not None) else obj_sp
                    goal_rel = (active.astype(np.float32) - ee)
                    if chunk is None or ci >= args.replan:
                        wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                        o_in = build_obs(np.asarray(obs["agentview_image"]), wr_raw,
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                        # log a distillation sample at each inference step
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        WR.append(image_tools.convert_to_uint8(wr)); GR.append(goal_rel.copy())
                        PR.append(prop); HE.append(int(held)); CH.append(chunk.copy())
                    a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_cnt = close_cnt + 1 if grip > 0 else 0
                    if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                    obs, _, done, _ = env.step(a[:7].tolist())
                    lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                env.close()
                if ok and len(WR) > 3:
                    n_ok += 1; n_samp += len(WR)
                    np.savez_compressed(out / f"{stem[:30]}_s{seed}_t{t}.npz",
                                        wrist=np.asarray(WR, np.uint8), goal_rel=np.asarray(GR, np.float32),
                                        proprio=np.asarray(PR, np.float32), held=np.asarray(HE, np.int32),
                                        chunk=np.asarray(CH, np.float32))
                print(f"  {stem[:34]:36s} s{seed} t{t} {'OK' if ok else '..'} samp={len(WR)} (tot {n_ok} rollouts/{n_samp} samp)", flush=True)
    print(f"\n=== COLLECTED {n_ok} successful rollouts, {n_samp} distill samples -> {out} ===", flush=True)
    print("COLLECT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
