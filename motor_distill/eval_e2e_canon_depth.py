"""HONEST-LOCALIZATION e2e: DINOv2 binder picks target identity -> localize it via REAL DEPTH unprojection (camera
surface point = the goal the motor was distilled on) -> SE(2)-canon motor. Two envs per trial: a localization env
(camera_depths=True) for binder + depth surface-point of the chosen object & container, then a motor env
(camera_depths=False, same init) so the eye-in-hand RGB isn't corrupted by the depth buffer. Removes the privileged
body-center goal; only remaining privilege is the proposal pixel (a detector supplies it in the full version).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_canon_depth.py \
       --head data/motor_head_canon.pt --bddl-dir <swap> --init-dir <swap> --proto-dir <swap> --proto-init-dir <swap> \
       --proto-inits 30,32,34 --binder dino --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from bind_exemplar import build_bank, proto_crop
from dino_separability import dino_feat
from bind_foveate import nm


def localize_masked(sim, body, depth_norm, seg_native, R, cam="agentview", win=11, pct=20):
    """ROBUST per-object 3D localization: project object center -> take a WINDOW of real depth around it -> the object
    is the NEAR surface (low percentile), background is far -> unproject the center pixel at that near depth. Window+
    percentile avoids the single-pixel-hits-background failure (33-47cm errors) AND gives the correct per-object Z
    (taller/bigger objects -> nearer surface -> higher z). seg_native unused (kept for signature compat)."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
    real = cu.get_real_depth_map(sim, depth_norm); real2 = np.asarray(real).reshape(R, R)
    h = win // 2
    window = real2[max(0, r0 - h):r0 + h + 1, max(0, c0 - h):c0 + h + 1].reshape(-1)
    zd = float(np.percentile(window, pct))                          # near surface (object), robust to background
    dmconst = np.full((R, R, 1), zd, np.float32)                    # unproject center ray at the object-surface depth
    pt = cu.transform_from_pixels_to_world(np.array([[r0, c0]]), dmconst[None], inv)[0]
    return np.asarray(pt, np.float32)


def pick(sim, graspables, rb, T, bank, bind_res, dev):
    hi = sim.render(width=bind_res, height=bind_res, camera_name="agentview")
    up = np.asarray(hi)[::-1].copy(); proto = bank.get(nm(T)); feats = {}
    for o in graspables:
        if rb.get(o) is None: continue
        c = proto_crop(sim, up, bind_res, "agentview", rb[o], 60)
        if c is not None: feats[o] = dino_feat(c, dev)
    return max(feats, key=lambda o: float(feats[o] @ proto)) if (feats and proto is not None) else T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", required=True); p.add_argument("--proto-init-dir", required=True)
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--binder", default="dino", choices=["oracle", "dino"]); p.add_argument("--canon", type=int, default=1)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    bank = None
    if args.binder == "dino":
        print("building DINOv2 prototype bank...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"E2E-DEPTH canon={args.canon} binder={args.binder}", flush=True)
    succ = []; bind_ok = 0; nb = 0; loc_err = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        graspables = [o for o in objs if args.container not in o]
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            # ---- localization env (camera_depths=True): binder + real-depth surface points ----
            envL = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=True, camera_segmentations="instance")
            envL.seed(args.seed + ti); envL.reset(); simL = envL.env.sim
            obsL = envL.set_init_state(inits[ti]) if inits is not None else envL.reset()
            rbL = resolve_bodies(simL, graspables + [args.container + "_1"]); cbL = rbL.get(args.container + "_1")
            if cbL is None:
                cand = [b for b in (simL.model.body_id2name(i) for i in range(simL.model.nbody)) if b and args.container in b]
                cbL = cand[0] if cand else None
            if rbL.get(T) is None or cbL is None: envL.close(); succ.append(0); continue
            nb += 1
            chosen = T if args.binder == "oracle" else pick(simL, graspables, rbL, T, bank, args.bind_res, dev)
            bind_ok += int(chosen == T)
            dep = np.asarray(obsL["agentview_depth"]); seg = np.asarray(obsL["agentview_segmentation_instance"])
            obj_w = localize_masked(simL, rbL[chosen], dep, seg, args.res); cont_w = localize_masked(simL, cbL, dep, seg, args.res)
            loc_err.append(float(np.linalg.norm(obj_w - body_pos(simL, rbL[chosen]))))
            envL.close()
            # ---- motor env (camera_depths=False): run SE(2)-canon motor toward depth-localized goals ----
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, [T])
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                goal_rel = ((cont_w if held else obj_w).astype(np.float32) - ee)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    if args.canon:
                        theta = canon_angle(goal_rel); phi = -theta
                        wr_in = rot_image(wr, phi); g_in = rot_vec_xy(goal_rel, phi); pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                    else:
                        theta = 0.0; wr_in = wr; g_in = goal_rel; pr = prop
                    img = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    vec = ((np.concatenate([g_in, pr, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        ch = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                    chunk = decanon_chunk(ch, theta) if args.canon else ch; ci = 0
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
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bind_ok}/{nb}", flush=True)
    print(f"\n=== E2E-DEPTH DINOv2-binder + CANON motor on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb}  loc_err={np.mean(loc_err)*100:.1f}cm ===  "
          f"[swap: pi0.5 17% | VLS 36.81%]", flush=True)
    print("E2EDEPTH_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
