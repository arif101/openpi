"""STAGE 1: ANALYTIC CLOSED-LOOP SERVO on the position-shift curve. No learned motor, no pi0, no memorization --
a scripted OSC waypoint controller that re-reads the LIVE object/basket position each step and position-servos to
it (above->descend->grasp->lift->transport->release). Run the SAME displacement curve as the distilled head and pi0.5.
If this FLATTENS (tracks pi0.5, beats our distilled head's decay), it proves: given correct LIVE localization, a
closed-loop servo is position-GENERAL by construction -> the motor is NOT the bottleneck, LOCALIZATION is (the
Phase-1 thesis), and we have a general pi0-FREE grasp controller. Official env metric. camera_depths=False (no depth).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_servo_posshift.py \
       --bddl-dir <std libero_object> --init-dir <...> --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from eval_wristcam_motor import geom_ids_for_body


def displace_object(sim, body_name, d, rng):
    bid = sim.model.body_name2id(body_name)
    for j in range(sim.model.njnt):
        if sim.model.jnt_bodyid[j] == bid and sim.model.jnt_type[j] == 0:
            adr = sim.model.jnt_qposadr[j]
            ang = rng.uniform(0, 2 * np.pi); sim.data.qpos[adr] += d * np.cos(ang); sim.data.qpos[adr + 1] += d * np.sin(ang)
            sim.forward(); return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--container", default="basket"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--trials", type=int, default=3); ap.add_argument("--init-start", type=int, default=30)
    ap.add_argument("--offsets", default="0,0.06,0.12,0.16"); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--K", type=float, default=15.0); ap.add_argument("--conv", type=float, default=0.02)
    ap.add_argument("--maxstep", type=int, default=360); ap.add_argument("--hide", type=int, default=0)
    ap.add_argument("--seed", type=int, default=5); ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--grasp-conv", type=float, default=0.012); ap.add_argument("--grasp-z", type=float, default=-0.005)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    H = args.res; ABOVE = 0.12
    offsets = [float(x) for x in args.offsets.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"ANALYTIC SERVO posshift  offsets(m)={offsets}  init-start={args.init_start}  hide={args.hide}", flush=True)
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
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H)
                env.seed(args.seed + ti); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); succ.append(0); reach.append(0); continue
                if args.hide:
                    rbd = resolve_bodies(sim, distract)
                    for o in distract:
                        if rbd.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rbd[o]): sim.model.geom_rgba[g, 3] = 0.0
                rng = np.random.default_rng(1000 * int(d * 1000) + ti)
                if d > 0: displace_object(sim, rb[T], d, rng)
                bsk = body_pos(sim, cb).astype(np.float32); z0 = body_pos(sim, rb[T])[2]
                if args.verbose:
                    o0 = body_pos(sim, rb[T]); ee0 = np.asarray(obs["robot0_eef_pos"], np.float32)
                    print(f"    [start {stem[:16]}] cb={cb} obj={np.round(o0,3)} basket={np.round(bsk,3)} ee={np.round(ee0,3)}", flush=True)
                wi = 0; hold = 0; lifted = 0.0; min_xy = 9.9
                for step in range(args.maxstep):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    obj = body_pos(sim, rb[T]).astype(np.float32)              # LIVE re-read (closed loop)
                    if wi < 2: min_xy = min(min_xy, float(np.linalg.norm(obj[:2] - ee[:2])))
                    gz = args.grasp_z
                    wps = [(obj + [0, 0, ABOVE], -1.0), (obj + [0, 0, gz], -1.0), (obj + [0, 0, gz], +1.0),
                           (obj + [0, 0, ABOVE], +1.0), (bsk + [0, 0, ABOVE + 0.05], +1.0),
                           (bsk + [0, 0, 0.08], +1.0), (bsk + [0, 0, 0.08], -1.0)]
                    wp, grip = np.asarray(wps[wi][0], np.float32), wps[wi][1]
                    a = np.zeros(7, np.float32); a[:3] = np.clip(args.K * (wp - ee), -1.0, 1.0); a[6] = grip
                    dconv = float(np.linalg.norm(ee - wp))
                    thr = args.grasp_conv if wi == 1 else args.conv      # tighter convergence before closing the gripper
                    if dconv < thr or (wi in (2, 6) and hold > 12):
                        hold += 1
                        if (wi in (2, 6) and hold > 12) or (wi not in (2, 6)): wi = min(wi + 1, len(wps) - 1); hold = 0
                    elif wi in (2, 6): hold += 1
                    obs, _, done, _ = env.step(a.tolist())
                    lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                    if done: break
                for _ in range(12): obs, _, done, _ = env.step([0,0,0,0,0,0,-1.0])  # settle (gripper open hold)
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                o2b = float(np.linalg.norm(body_pos(sim, rb[T])[:2] - bsk[:2]))
                if args.verbose:
                    print(f"    [{stem[:20]:22s} d={d*100:.0f}] lifted={lifted*100:4.1f}cm obj2basket={o2b*100:5.1f}cm "
                          f"succ={int(ok)}", flush=True)
                succ.append(int(ok)); reach.append(int(min_xy < 0.03)); env.close()
        curve[d] = float(np.mean(succ)) if succ else 0.0
        rr = float(np.mean(reach)) if reach else 0.0
        print(f"  d={d*100:4.0f}cm  success={curve[d]:.3f}  reach<3cm={rr:.3f}  (n={len(succ)})", flush=True)
    print("\n=== ANALYTIC CLOSED-LOOP SERVO position-shift curve (no pi0, no learned motor) ===", flush=True)
    for d in offsets: print(f"   {d*100:5.0f} cm : {curve[d]:.3f}", flush=True)
    print(f"   delta = {curve[offsets[-1]]-curve[offsets[0]]:+.3f}  [flat => closed-loop servo is position-GENERAL]", flush=True)
    print("SERVOPOSSHIFT_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
