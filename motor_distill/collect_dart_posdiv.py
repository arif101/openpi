"""GOAL-CORRECTING build: DART recovery data at DIVERSE positions. This is the key experiment distinguishing
'goal-correcting' from 'goal-following'. The earlier null (adding successful on-track swap demos didn't help) was
because successful demos can't teach CORRECTION. Here we (1) DISPLACE the target by a continuous random offset
(diverse positions), (2) HIDE distractors so pi0.5 still succeeds, (3) PERTURB the executed action so the gripper
drifts OFF-goal, and (4) log pi0.5's CORRECTIVE chunk from those off-goal states. Result = (wrist, goal_rel, proprio,
held -> pi0.5 correction) pairs spanning off-goal states AND diverse positions -> trains a position-general
goal-CORRECTING wrist motor. Target: the posshift curve tracks pi0.5 (close the 12cm gap 0.30->~0.83). pi0.5 offline only.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/collect_dart_posdiv.py \
       --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object \
       --init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object \
       --out data/motor_demos_dart --trials 12 --seeds 0,1 --max-disp 0.10 --dart-noise 0.02 --dart-prob 0.35
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs, _quat2axisangle
from collect_motor_data import surface_point
from collect_motor_data_hide import geom_ids_for_body


def displace_object(sim, body_name, d, rng):
    bid = sim.model.body_name2id(body_name)
    for j in range(sim.model.njnt):
        if sim.model.jnt_bodyid[j] == bid and sim.model.jnt_type[j] == 0:
            adr = sim.model.jnt_qposadr[j]
            ang = rng.uniform(0, 2 * np.pi); sim.data.qpos[adr] += d * np.cos(ang); sim.data.qpos[adr + 1] += d * np.sin(ang)
            sim.forward(); return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--out", default="data/motor_demos_dart"); p.add_argument("--container", default="basket")
    p.add_argument("--trials", type=int, default=12); p.add_argument("--seeds", default="0,1")
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--max-disp", type=float, default=0.10)   # continuous random object displacement up to this (m)
    p.add_argument("--hide", type=int, default=1)
    p.add_argument("--dart-noise", type=float, default=0.02); p.add_argument("--dart-prob", type=float, default=0.35)
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
    print(f"{len(bddls)} tasks seeds={seeds} trials={args.trials} max-disp={args.max_disp} hide={args.hide}", flush=True)
    n_ok = 0; n_try = 0; n_samp = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        distract = [o for o in objs if args.container not in o and o != T]
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        for seed in seeds:
            nt = min(args.trials, len(inits)) if inits is not None else args.trials
            for t in range(nt):
                n_try += 1
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
                env.seed(seed + t); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[t]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T, args.container + "_1"] + distract); cb = rb[args.container + "_1"]
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); continue
                if args.hide:
                    for o in distract:
                        if rb.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rb[o]): sim.model.geom_rgba[g, 3] = 0.0
                rng = np.random.default_rng(7919 * (seed + 1) + t)
                d = float(rng.uniform(0, args.max_disp))
                if d > 0:
                    displace_object(sim, rb[T], d, rng)
                    try: obs = env.env._get_observations(force_update=True)
                    except Exception: pass
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
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        WR.append(image_tools.convert_to_uint8(wr)); GR.append(goal_rel.copy())
                        PR.append(prop); HE.append(int(held)); CH.append(chunk.copy())
                    a = chunk[ci].copy(); ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_cnt = close_cnt + 1 if grip > 0 else 0
                    if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                    if args.dart_noise > 0 and (rng.random() < args.dart_prob) and not held:
                        a[:3] = a[:3] + rng.normal(0, args.dart_noise, 3).astype(np.float32)
                        ci = args.replan   # force re-plan from perturbed (off-goal) state -> pi0.5 CORRECTION
                    obs, _, done, _ = env.step(a[:7].tolist())
                    lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                env.close()
                if ok and len(WR) > 3:
                    n_ok += 1; n_samp += len(WR)
                    np.savez_compressed(out / f"dart_{stem[:26]}_s{seed}_t{t}.npz",
                                        wrist=np.asarray(WR, np.uint8), goal_rel=np.asarray(GR, np.float32),
                                        proprio=np.asarray(PR, np.float32), held=np.asarray(HE, np.int32),
                                        chunk=np.asarray(CH, np.float32))
                print(f"  {stem[:30]:32s} s{seed} t{t} d={d*100:4.1f}cm {'OK' if ok else '..'} "
                      f"samp={len(WR)} (ok {n_ok}/{n_try}, {n_samp} samp)", flush=True)
    print(f"\n=== DART-POSDIV collected {n_ok}/{n_try} rollouts, {n_samp} CORRECTION samples -> {out} ===", flush=True)
    print("DARTPOSDIV_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
