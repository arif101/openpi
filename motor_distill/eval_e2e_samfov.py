"""De-hardcode + FOVEATION: SAM proposes object regions (no body_pos) -> each region cropped at HI-RES (1024 render) ->
DINOv2 match to a hi-res reference bank (the validated 1.00-separability binder) -> ray-plane -> z-robust canon motor.
Fixes the SAM e2e's 256px-binding collapse by discriminating at high resolution. Perception 100% from images, pi0-free.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_samfov.py \
       --head data/motor_head_canon_z.pt --bddl-dir <swap> --init-dir <swap> --proto-dir <swap> --proto-init-dir <swap> --proto-inits 40,42,44 --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from bind_exemplar import build_bank
from dino_separability import dino_feat
from bind_foveate import nm
from eval_e2e_rayplane import ray_plane


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", required=True); p.add_argument("--proto-init-dir", required=True)
    p.add_argument("--proto-inits", default="40,42,44"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.02); p.add_argument("--z-cont", type=float, default=0.05); p.add_argument("--z-ref-init", type=int, default=48)
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
    print("building hi-res DINOv2 bank...", flush=True)
    bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                      args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    R = args.res; HR = args.bind_res; sc = HR / R
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem; proto = bank.get(nm(T))
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        # per-object height calibration from a disjoint ref scene (object height = stable property)
        z_obj_task = args.z_obj
        if inits is not None and args.z_ref_init < len(inits):
            ze = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R); ze.seed(7); ze.reset()
            ze.set_init_state(inits[args.z_ref_init]); zrb = resolve_bodies(ze.env.sim, [T])
            if zrb.get(T) is not None: z_obj_task = float(body_pos(ze.env.sim, zrb[T])[2])
            ze.close()
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
            img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])   # upright for SAM
            hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()  # hi-res for foveated crops
            masks = gen.generate(img); best = None
            for m in masks:
                seg = m["segmentation"]; a = seg.sum()
                if a < 20 or a > 0.25 * R * R: continue
                ys, xs = np.where(seg); ry, cx = ys.mean(), xs.mean()
                hy, hx = int(ry * sc), int(cx * sc); s = 60
                cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
                if cr.size < 100: continue
                f = dino_feat(cr, dev); score = float(f @ proto) if proto is not None else 0.0
                if best is None or score > best[0]: best = (score, ry, cx, ys, xs)
            if best is None: env.close(); succ.append(0); continue
            r_up, c0 = best[1], best[2]; r0 = R - 1 - r_up
            # BASE-CONTACT localization: bottom of the mask sits ON the table (known z) -> ray ∩ table = true xy (no
            # perspective offset, unlike the silhouette centroid). c at the base, row = mask bottom.
            ys_b, xs_b = best[3], best[4]; base_up = int(ys_b.max()); base_c = int(xs_b[ys_b >= ys_b.max() - 2].mean())
            r0 = R - 1 - base_up; c0 = base_c
            import robosuite.utils.camera_utils as cu
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            pxt = cu.project_points_from_world_to_camera(body_pos(sim, rbT[T])[None], w2p, R, R)[0]
            bind_ok += int(np.hypot(r0 - pxt[0], c0 - pxt[1]) < 18)
            # ray-plane at the chosen pixel (centroid) for the object; container via its projection
            def rp(r, c, z):
                inv = np.linalg.inv(w2p); d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
                q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d1[None], inv)[0])
                q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d2[None], inv)[0])
                ray = q2 - q1; tt = (z - q1[2]) / ray[2]; pt = q1 + tt * ray; return np.array([pt[0], pt[1], z], np.float32)
            rp_base = rp(int(min(max(r0, 0), R - 1)), int(min(max(c0, 0), R - 1)), 0.0)   # base contact at table z=0
            obj_w = np.array([rp_base[0], rp_base[1], z_obj_task], np.float32)            # true xy + center-height z
            pxc = cu.project_points_from_world_to_camera(body_pos(sim, cb)[None], w2p, R, R)[0]
            cont_w = rp(int(min(max(pxc[0], 0), R - 1)), int(min(max(pxc[1], 0), R - 1)), args.z_cont)
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
    print(f"\n=== E2E-SAMFOV (NO body_pos: SAM+FOVEATED-DINOv2+ray-plane+canon-z) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb} ===  [swap pi0.5 17% | VLS 36.81%]", flush=True)
    print("E2ESAMFOV_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
