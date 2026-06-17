"""Closed-loop eval of the FLOW-MATCHING wrist motor (precision-preserving distillation). Same harness as
eval_posshift (object displacement curve, surface-goal 0 = body-center privileged goal, optional hide, official
metric, camera_depths=False to avoid the EGL inference-corruption bug). Each replan: sample an action chunk from the
flow head. Compare to the L1-regression head (standard~0.73@init30, swap 0.50, posshift 0.73/0.63/0.30/0.33).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_flowmotor.py \
       --head data/flowmotor.pt --bddl-dir <suite> --init-dir <...> --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_flowmotor import FlowMotor
from eval_wristcam_motor import _surface_point, geom_ids_for_body
from eval_posshift import displace_object


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/flowmotor.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=30); p.add_argument("--res", type=int, default=256)
    p.add_argument("--offsets", default="0"); p.add_argument("--surface-goal", type=int, default=0)
    p.add_argument("--hide", type=int, default=0); p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = FlowMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]; cm = torch.tensor(ck["cm"], device=dev); cs = torch.tensor(ck["cs"], device=dev)
    offsets = [float(x) for x in args.offsets.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"FLOW motor  head={args.head}  offsets={offsets}  init-start={args.init_start}  hide={args.hide}", flush=True)
    curve = {}
    for d in offsets:
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
            s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
            for t in range(nt):
                ti = s0 + t
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=bool(args.surface_goal))
                env.seed(args.seed + ti); env.reset(); sim = env.env.sim
                obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
                rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
                if cb is None:
                    cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                    cb = cand[0] if cand else None
                if rb[T] is None or cb is None: env.close(); succ.append(0); continue
                if args.hide:
                    distract = [o for o in objs if args.container not in o and o != T]
                    rbd = resolve_bodies(sim, distract)
                    for o in distract:
                        if rbd.get(o) is None: continue
                        for g in geom_ids_for_body(sim, rbd[o]): sim.model.geom_rgba[g, 3] = 0.0
                    obs = env.set_init_state(inits[ti]) if inits is not None else obs
                rng = np.random.default_rng(1000 * int(d * 1000) + ti)
                if d > 0:
                    displace_object(sim, rb[T], d, rng)
                    try: obs = env.env._get_observations(force_update=True)
                    except Exception: pass
                z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
                osp = _surface_point(sim, rb[T], np.asarray(obs["agentview_depth"]), args.res) if args.surface_goal else None; csp = None
                for step in range(args.horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                    if args.surface_goal:
                        if held and csp is None: csp = _surface_point(sim, cb, np.asarray(obs["agentview_depth"]), args.res)
                        active = (csp if (held and csp is not None) else osp)
                    else:
                        active = body_pos(sim, cb) if held else body_pos(sim, rb[T])
                    goal_rel = (active.astype(np.float32) - ee)
                    if chunk is None or ci >= args.replan:
                        wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                        img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        vec = np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32)
                        vec = ((vec - vm) / vs).astype(np.float32)
                        with torch.no_grad():
                            ch = net.sample(img, torch.tensor(vec)[None].to(dev), steps=args.steps)
                            chunk = (ch.reshape(1, -1) * cs + cm).reshape(net.chunk, net.act).cpu().numpy(); ci = 0
                    a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_cnt = close_cnt + 1 if grip > 0 else 0
                    if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                    act = a[:7].copy(); act[6] = grip
                    obs, _, done, _ = env.step(act.tolist())
                    lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                succ.append(int(ok)); env.close()
        curve[d] = float(np.mean(succ)) if succ else 0.0
        print(f"  d={d*100:4.0f}cm  success={curve[d]:.3f}  (n={len(succ)})", flush=True)
    print(f"\n=== FLOW-MOTOR ({pathlib.Path(args.bddl_dir).name}) ===", flush=True)
    for d in offsets: print(f"   {d*100:5.0f} cm : {curve[d]:.3f}", flush=True)
    print("FLOWEVAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
