"""STAGE 3 (eval) of the factored wrist-cam motor. Run the trained IDENTITY-AGNOSTIC head CLOSED-LOOP, conditioned
on a PRIVILEGED correct relative goal (object before grasp, container after) -> isolates MOTOR PRECISION from
binding (same philosophy as the pi0.5 ceiling test). Torch only (no pi0.5/jax).

Compare points:
  - blind goal-relative motor (no wrist):      16.7%  (prior)
  - pi0.5 frozen-policy privileged STEERING:   30%    (swap_priv; crude, fragile)
  - pi0.5 full motor (both cams, correct bind): ~100%  (ceiling)
HEADLINE = run on the SWAP axis: a position-AGNOSTIC motor given a correct goal should NOT suffer the relocation
collapse (pi0.5 swap=17%). If our head >> 30%, learning-to-use-goal beats steering and we have a position-robust motor.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_wristcam_motor.py \
       --head data/motor_head.pt --bddl-dir <dir> --init-dir <dir> --container basket --n 10 --trials 3
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor


def _surface_point(sim, body, depth_norm, R, cam="agentview"):
    import robosuite.utils.camera_utils as cu
    pos = body_pos(sim, body); w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
    r = int(min(max(px[0], 0), R - 1)); c = int(min(max(px[1], 0), R - 1))
    real = cu.get_real_depth_map(sim, depth_norm)
    return np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), real[None], np.linalg.inv(w2p))[0], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt")
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=0); p.add_argument("--res", type=int, default=256)
    p.add_argument("--goal-offset", type=float, default=0.0)   # sim binder error: fixed random offset per rollout
    p.add_argument("--surface-goal", type=int, default=0)   # use binder-style surface-point goal (needs depth)   # held-out: index into inits[] so eval positions != training positions
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
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
        s0 = args.init_start
        nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
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
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            osp = _surface_point(sim, rb[T], np.asarray(obs["agentview_depth"]), args.res) if args.surface_goal else None; csp = None
            if args.goal_offset > 0:
                _o = np.random.default_rng(args.seed + ti).normal(size=3); _o = (_o/np.linalg.norm(_o))*args.goal_offset
            else: _o = np.zeros(3, np.float32)
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                if args.surface_goal:
                    if held and csp is None: csp = _surface_point(sim, cb, np.asarray(obs["agentview_depth"]), args.res)
                    active = (csp if (held and csp is not None) else osp)
                else:
                    active = body_pos(sim, cb) if held else body_pos(sim, rb[T])
                goal_rel = (active.astype(np.float32) - ee) + _o.astype(np.float32)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                    img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    vec = np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32)
                    vec = ((vec - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        chunk = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]; ci = 0
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
        print(f"  {stem[:34]:36s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== WRIST-CAM MOTOR (privileged goal, {pathlib.Path(args.bddl_dir).name}): {np.mean(succ):.3f} "
          f"({sum(succ)}/{len(succ)}) ===  [blind 16.7% | steering 30% | pi0.5 ceiling ~100%]", flush=True)
    print("EVAL_MOTOR_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
