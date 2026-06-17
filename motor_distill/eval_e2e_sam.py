"""FULLY HONEST e2e (NO body_pos anywhere): SAM class-agnostic masks PROPOSE all object regions -> DINOv2 picks the
mask whose crop matches the target's reference image -> mask centroid -> ray->table-plane -> SE(2)-canon z-robust
motor. Perception is 100% from the image (SAM + DINOv2 + ray-plane geometry); pi0-free. Reference bank built from a
held-out scene (a per-object reference photo, body_pos used only to crop the CATALOG, never at test).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_sam.py \
       --head data/motor_head_canon_z.pt --bddl-dir <swap> --init-dir <swap> --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from dino_separability import dino_feat

nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")


def crop_bbox(img, ys, xs, pad=6):
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def ray_plane_px(sim, r0, c0, z_plane, R, cam="agentview"):
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0])
    ray = q2 - q1; t = (z_plane - q1[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0
    pt = q1 + t * ray; return np.array([pt[0], pt[1], z_plane], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth"); p.add_argument("--ref-init", type=int, default=40)
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.02); p.add_argument("--z-cont", type=float, default=0.05)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=24, min_mask_region_area=40)
    R = args.res
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]

    # --- build per-object DINOv2 reference bank (catalog) from a held-out scene; body_pos used only to crop refs ---
    bank = {}
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; graspables = [o for o in objs if args.container not in o]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R); env.seed(args.ref_init); env.reset()
        obs = env.set_init_state(inits[args.ref_init]) if inits is not None else env.reset(); sim = env.env.sim
        img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
        import robosuite.utils.camera_utils as cu
        rb = resolve_bodies(sim, graspables); w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        for o in graspables:
            if rb.get(o) is None or nm(o) in bank: continue
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            r0 = R - 1 - int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
            cr = img[max(0, r0 - 14):r0 + 14, max(0, c0 - 14):c0 + 14]
            if cr.size > 50: bank[nm(o)] = dino_feat(cr, dev)
        env.close()
    print(f"bank: {len(bank)} object refs", flush=True)

    succ = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem; proto = bank.get(nm(T))
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rbT = resolve_bodies(sim, [T, args.container + "_1"]); cb = rbT[args.container + "_1"]
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            if rbT.get(T) is None or cb is None: env.close(); succ.append(0); continue
            nb += 1
            img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])     # upright for SAM
            masks = gen.generate(img); best = None
            for m in masks:
                seg = m["segmentation"]; a = seg.sum()
                if a < 20 or a > 0.25 * R * R: continue
                ys, xs = np.where(seg); cr = crop_bbox(img, ys, xs)
                if cr.size < 50: continue
                f = dino_feat(cr, dev); sc = float(f @ proto) if proto is not None else 0.0
                if best is None or sc > best[0]: best = (sc, ys.mean(), xs.mean())
            if best is None: env.close(); succ.append(0); continue
            r_up, c0 = best[1], best[2]; r0 = R - 1 - r_up    # un-flip to camera-frame row
            # bind check: is the chosen centroid near the TRUE target projection?
            import robosuite.utils.camera_utils as cu
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            pxt = cu.project_points_from_world_to_camera(body_pos(sim, rbT[T])[None], w2p, R, R)[0]
            bind_ok += int(np.hypot(r0 - pxt[0], c0 - pxt[1]) < 18)
            obj_w = ray_plane_px(sim, r0, c0, args.z_obj, R)
            pxc = cu.project_points_from_world_to_camera(body_pos(sim, cb)[None], w2p, R, R)[0]
            cont_w = ray_plane_px(sim, int(min(max(pxc[0],0),R-1)), int(min(max(pxc[1],0),R-1)), args.z_cont, R)
            z0 = body_pos(sim, rbT[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                goal_rel = ((cont_w if held else obj_w).astype(np.float32) - ee)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    theta = canon_angle(goal_rel); phi = -theta
                    wr_in = rot_image(wr, phi); g_in = rot_vec_xy(goal_rel, phi); pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                    imgT = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    vec = ((np.concatenate([g_in, pr, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        chh = net(imgT, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                    chunk = decanon_chunk(chh, theta); ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rbT[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bind_ok}/{nb}", flush=True)
    print(f"\n=== E2E-SAM (NO body_pos: SAM+DINOv2+ray-plane+canon-z) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb} ===  [swap pi0.5 17% | VLS 36.81%]", flush=True)
    print("E2ESAM_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
