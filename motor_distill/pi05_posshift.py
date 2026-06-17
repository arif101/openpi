"""ARCHITECTURE-VALIDATING experiment: run the SAME position-shift extrapolation curve on pi0.5+hide (a known
near-perfect GOAL-CORRECTING motor) and compare to our goal-FOLLOWING wrist motor. Displace the target by
continuous d, hide distractors so pi0.5 targets the (displaced) visible object. If pi0.5's curve is FLAT while ours
decays -> 'goal-correcting => position-general' is PROVEN and the goal-correcting servo build is justified. If pi0.5
also decays -> displacement is intrinsically hard (reachability) and we rethink.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_posshift.py \
       --bddl-dir <std libero_object> --init-dir <...> --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def geom_ids_for_body(sim, body_name):
    bid = sim.model.body_name2id(body_name)
    return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]


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
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--res", type=int, default=256)
    p.add_argument("--offsets", default="0,0.06,0.12,0.16"); p.add_argument("--init-start", type=int, default=30)
    p.add_argument("--hide", type=int, default=1)
    args = p.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    offsets = [float(x) for x in args.offsets.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"pi0.5+hide posshift  offsets(m)={offsets}  init-start={args.init_start}  hide={args.hide}", flush=True)
    curve = {}
    for d in offsets:
        succ = []; reach = []
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
            s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
            for t in range(nt):
                ti = s0 + t
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res)
                env.seed(ti); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T] + distract)
                if rb.get(T) is None: env.close(); succ.append(0); reach.append(0); continue
                if args.hide:
                    for o in distract:
                        if rb.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rb[o]): sim.model.geom_rgba[g, 3] = 0.0
                rng = np.random.default_rng(1000 * int(d * 1000) + ti)
                if d > 0:
                    displace_object(sim, rb[T], d, rng)
                    try: obs = env.env._get_observations(force_update=True)
                    except Exception: pass
                z0 = body_pos(sim, rb[T])[2]; min_xy = 9.9; chunk = None; ci = 0
                for step in range(args.horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    min_xy = min(min_xy, float(np.linalg.norm(body_pos(sim, rb[T])[:2] - ee[:2])))
                    if chunk is None or ci >= args.replan:
                        o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                    obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                succ.append(int(ok)); reach.append(int(min_xy < 0.03)); env.close()
        curve[d] = float(np.mean(succ)) if succ else 0.0
        rr = float(np.mean(reach)) if reach else 0.0
        print(f"  d={d*100:4.0f}cm  success={curve[d]:.3f}  reach<3cm={rr:.3f}  (n={len(succ)})", flush=True)
    print("\n=== pi0.5+hide POSITION-SHIFT CURVE (goal-CORRECTING reference) ===", flush=True)
    for d in offsets: print(f"   {d*100:5.0f} cm : {curve[d]:.3f}", flush=True)
    print(f"   delta = {curve[offsets[-1]]-curve[offsets[0]]:+.3f}  [flat => goal-correcting IS position-general]", flush=True)
    print("PI05POSSHIFT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
