"""Rung 0 (research recommendation): proper scripted OSC waypoint controller with TRUE goals -> proves the
motor GEOMETRY is solvable and gives the ORACLE CEILING the learned head is measured against. This is NOT the
deliverable (the learned chunked head is) -- it's the sanity/ceiling baseline that answers 'is OSC the problem
or our weak distillation?'. Correct controller this time: moving waypoint setpoints, position servo, advance
on convergence, gripper per waypoint. (Orientation held at the start top-down pose; add ori-servo if grasp fails.)

Waypoints: above-obj -> descend-to-obj -> close -> lift -> above-basket -> descend -> open.
--use-bound: use foveation-bound goals instead of true (tests binder precision vs the oracle ceiling).

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/oracle_osc.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--container", default="basket"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--K", type=float, default=15.0); ap.add_argument("--conv", type=float, default=0.02)
    ap.add_argument("--place-cm", type=float, default=12.0); ap.add_argument("--maxstep", type=int, default=400)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    H = args.res; PLACE = args.place_cm / 100; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, [T, args.container + "_1"]); cbody = rb.get(args.container + "_1")
        except Exception:
            env.close(); continue
        obj = body_pos(sim, rb[T]).astype(np.float32); bsk = body_pos(sim, cbody).astype(np.float32)
        z0 = obj[2]
        ABOVE = 0.12
        # waypoint = (target_xyz, gripper)  gripper: -1 open, +1 close
        wps = [
            (obj + [0, 0, ABOVE], -1.0),     # 0 above object
            (obj + [0, 0, 0.01], -1.0),      # 1 descend onto object
            (obj + [0, 0, 0.01], +1.0),      # 2 close (grasp)
            (obj + [0, 0, ABOVE], +1.0),     # 3 lift
            (bsk + [0, 0, ABOVE + 0.05], +1.0),  # 4 above basket
            (bsk + [0, 0, 0.08], +1.0),      # 5 descend into basket
            (bsk + [0, 0, 0.08], -1.0),      # 6 release
        ]
        wps = [(np.asarray(p, np.float32), g) for p, g in wps]
        wi = 0; hold = 0; lifted = 0.0
        for step in range(args.maxstep):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            wp, grip = wps[wi]
            a = np.zeros(7, np.float32)
            a[:3] = np.clip(args.K * (wp - ee), -1.0, 1.0)       # position servo to waypoint
            a[3:6] = 0.0                                          # hold start (top-down) orientation
            a[6] = grip
            d = float(np.linalg.norm(ee - wp))
            # advance: position converged (and for the close/open steps, give the gripper time to actuate)
            if d < args.conv or (wi in (2, 6) and hold > 12):
                hold += 1
                if (wi in (2, 6) and hold > 12) or (wi not in (2, 6)):
                    wi = min(wi + 1, len(wps) - 1); hold = 0
            elif wi in (2, 6):
                hold += 1
            obs, _, done, _ = env.step(a.tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if wi >= len(wps) - 1 and step > 30:                 # released, let settle a few steps
                if step % 1 == 0 and step > 0:
                    pass
            if done: break
        for _ in range(15):                                      # settle
            obs, _, done, _ = env.step(np.array([0,0,0,0,0,0,-1.0]).tolist())
        cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
        objdist = float(np.linalg.norm(cT[:2] - cC[:2]))
        placed = bool(lifted > 0.04 and objdist < PLACE)
        succ.append(placed); env.close()
        print(f"  {nm(T):14s} lifted={lifted*100:3.0f}cm obj2basket={objdist*100:4.0f}cm placed={'Y' if placed else '.'}", flush=True)
    n = len(succ)
    print(f"\n=== ORACLE OSC waypoint controller + TRUE goals (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  FULL pick-place success: {sum(succ)}/{n} = {sum(succ)/n*100:.0f}%   (oracle ceiling; CAG bar 21.7%)", flush=True)
    print("ORACLE_OSC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
