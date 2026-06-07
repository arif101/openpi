"""END-TO-END (NO privileged goal): vision BINDER (open-vocab detect on agentview + depth unprojection -> 3D goal)
-> our IDENTITY/POSITION-AGNOSTIC wrist-cam MOTOR. This is the real LIBERO-PRO number to compare vs VLS 36.81%.

Binder: GroundingDINO detects the NAMED object (and the container) on the agentview RGB -> box center pixel ->
robosuite depth unprojection (get_real_depth_map + transform_from_pixels_to_world) -> 3D world position.
Motor: drives toward that 3D goal (object pre-grasp, container post-grasp) using only wrist-cam + relative-goal.

--validate-binder: measure predicted-vs-TRUE 3D error + detection rate (NO motor) -> proves the geometry before
we trust any e2e number. Run this FIRST.

Run (validate): PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_wristmotor.py \
       --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object --init-dir <...> --validate-binder --n 10 --trials 2
Run (e2e):      same but drop --validate-binder and add --head data/motor_head.pt
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, _quat2axisangle


def binder_3d(sim, rgb, depth_norm, name, device, cam, H, W):
    """Open-vocab detect `name` on rgb -> center pixel -> unproject via depth -> 3D world point (or None)."""
    import robosuite.utils.camera_utils as cu
    box = detect_box(rgb, "a " + name.replace("_", " "), device)
    if box is None:
        return None, None
    x0, y0, x1, y1 = box
    u = int(round((x0 + x1) / 2)); v = int(round((y0 + y1) / 2))   # col, row
    u = min(max(u, 0), W - 1); v = min(max(v, 0), H - 1)
    real_depth = cu.get_real_depth_map(sim, depth_norm)            # HxWx1 metric
    w2p = cu.get_camera_transform_matrix(sim, cam, H, W)           # world->pixel
    cam2world = np.linalg.inv(w2p)
    # transform wants pixels [...,2] (row,col) and depth_map [...,H,W,1] with matching leading dims
    pts = cu.transform_from_pixels_to_world(np.array([[v, u]]), real_depth[None], cam2world)
    return np.asarray(pts[0], np.float32), (u, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--cam", default="agentview"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--trials", type=int, default=3); p.add_argument("--horizon", type=int, default=280)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--seed", type=int, default=5)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--img", type=int, default=128)
    p.add_argument("--validate-binder", action="store_true")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = None
    if not args.validate_binder:
        from train_wristcam_motor import WristMotor
        ck = torch.load(args.head, map_location=dev, weights_only=False)
        net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
        vm, vs = ck["vm"], ck["vs"]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []; berr = []; ndet = 0; ntot = 0
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
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=True)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            H = W = 256
            if args.validate_binder:
                # measure binder 3D error for the target object vs ground truth
                rgb = np.asarray(obs[args.cam + "_image"]); dep = np.asarray(obs[args.cam + "_depth"])
                pred, px = binder_3d(sim, rgb, dep, T, dev, args.cam, H, W)
                ntot += 1
                if pred is not None:
                    ndet += 1; true = body_pos(sim, rb[T]); e = float(np.linalg.norm(pred - true)); berr.append(e)
                    print(f"  {stem[:30]:32s} det@{px} pred={np.round(pred,3)} true={np.round(true,3)} err={e*100:.1f}cm", flush=True)
                else:
                    print(f"  {stem[:30]:32s} NO DETECTION", flush=True)
                env.close(); continue
            # ----- full e2e: binder goal + motor -----
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            goal_w = None
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                if chunk is None or ci >= args.replan:
                    rgb = np.asarray(obs[args.cam + "_image"]); dep = np.asarray(obs[args.cam + "_depth"])
                    tgt_name = args.container if held else T
                    g, _ = binder_3d(sim, rgb, dep, tgt_name, dev, args.cam, H, W)
                    if g is not None: goal_w = g                       # keep last good detection
                    goal_rel = (goal_w - ee).astype(np.float32) if goal_w is not None else np.zeros(3, np.float32)
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
        if not args.validate_binder:
            print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    if args.validate_binder:
        med = np.median(berr) if berr else float("nan")
        print(f"\n=== BINDER VALIDATION: detect {ndet}/{ntot}, median 3D err = {med*100:.1f}cm "
              f"(mean {np.mean(berr)*100:.1f}cm) ===  [<5cm = good enough for the motor]", flush=True)
        print("BINDER_VALIDATE_EXIT=0", flush=True)
    else:
        print(f"\n=== E2E (vision binder + wrist motor, {pathlib.Path(args.bddl_dir).name}): {np.mean(succ):.3f} "
              f"({sum(succ)}/{len(succ)}) ===  [VLS 36.81% | pi0.5 baseline 23.69%]", flush=True)
        print("E2E_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
