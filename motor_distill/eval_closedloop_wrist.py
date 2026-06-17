"""CLOSED-LOOP eye-in-hand re-localization: the architecture that breaks the open-loop motor's <2cm goal requirement.
A coarse (~3cm) goal lands the object in the WRIST FOV (foundational test: 100% in-FOV when close); we then RE-GROUND
the goal on the wrist camera each replan (wrist ray-plane ~1cm, vs overhead ~3cm) -> the motor grasps. Reach-then-servo:
the coarse goal only has to get close; the wrist closes the last cm. This is what pi0.5 does implicitly (re-grounds on
cameras each step) and what IBVS guarantees (robust to a coarse target by construction).

Three arms isolate LOCALIZATION (oracle binding, so binding is not the variable):
  A  coarse goal (+--goal-noise), --relocalize none          -> expect collapse (the wall)
  B  coarse goal (+--goal-noise), --relocalize wrist          -> expect recovery (the fix)
  C  perfect goal (--goal-noise 0), --relocalize none         -> the 0.50 motor ceiling
Wrist pixel: --wrist-src oracle = object projected into the wrist cam (architecture upper bound); honest detector swaps in.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_closedloop_wrist.py \
       --head data/motor_head_canon_z.pt --bddl-dir <swap> --init-dir <swap> --relocalize wrist --goal-noise 0.03 \
       --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from eval_e2e_rayplane import ray_plane
from bind_exemplar import build_bank, proto_crop
from dino_separability import dino_feat
from bind_foveate import nm
from diag_triangulate import pixel_ray, closest_point


def _ray_ray_dist(o1, d1, o2, d2):
    """Shortest distance between two skew rays (epipolar gating for cross-view correspondence)."""
    p = closest_point(o1, d1, o2, d2)
    # distance from p to each ray, summed
    return float(np.linalg.norm((p - o1) - ((p - o1) @ d1) * d1) + np.linalg.norm((p - o2) - ((p - o2) @ d2) * d2))


def wrist_ray_plane(sim, pix, z_plane, R, cam="robot0_eye_in_hand"):
    """Ray through a WRIST-cam pixel ∩ table plane z=z_plane. The wrist transform is recomputed each call (the camera
    moves with the arm). Returns world xy on the table plane."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    r0 = int(min(max(pix[0], 0), R - 1)); c0 = int(min(max(pix[1], 0), R - 1))
    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0])
    ray = q2 - q1; t = (z_plane - q1[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0
    pt = q1 + t * ray; return np.array([pt[0], pt[1], z_plane], np.float32)


def wrist_proj(sim, world_xyz, R, cam="robot0_eye_in_hand"):
    """Project a world point into the wrist cam; return (row,col) and in-FOV bool."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(np.asarray(world_xyz)[None], w2p, R, R)[0]
    infov = (0 <= px[0] < R) and (0 <= px[1] < R)
    return px, infov


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.05)
    p.add_argument("--z-ref-init", type=int, default=35)
    p.add_argument("--relocalize", default="wrist", choices=["none", "wrist"])
    p.add_argument("--wrist-src", default="oracle", choices=["oracle", "honest", "rebind"])
    p.add_argument("--rebind-thr", type=float, default=0.0)   # min DINOv2 identity score to accept a wrist re-bind
    p.add_argument("--av-oracle", type=int, default=0)        # diag: use true agentview pixel for the epipolar ray
    p.add_argument("--wrist-sel", default="epi", choices=["epi", "id_epi"])  # wrist mask selection: min-ray-ray vs DINOv2-identity gated by epipolar+size
    p.add_argument("--epi-thr", type=float, default=0.04)     # epipolar gate: max ray-ray distance (m) to consider a wrist mask
    p.add_argument("--wrist-center", default="centroid", choices=["centroid", "dt", "bbox"])  # object-center estimator on the selected wrist mask
    p.add_argument("--wrist-id", type=int, default=0)         # multi-view fusion: score wrist masks by WRIST-view DINOv2 protos + epipolar agreement
    p.add_argument("--epi-w", type=float, default=20.0)       # weight on epipolar agreement (per metre of ray-ray distance) in the fused wrist score
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--loc-method", default="base", choices=["base", "centroid"])  # agentview localization: mask base@z=0 vs centroid@z=obj-height
    p.add_argument("--wrist-flip", type=int, default=1)   # row -> R-1-row to match the camera matrix (render is bottom-origin)
    p.add_argument("--sel-radius", type=int, default=60)  # px: pick the SAM mask whose centroid is nearest the projected coarse goal
    p.add_argument("--goal-noise", type=float, default=0.03)   # coarse-localization error injected into the initial xy goal (noise mode)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--coarse-src", default="noise", choices=["noise", "dino", "sam", "triang", "depth"])  # depth = foveated id + DEPTH-median (honest ~3cm, breaks the silhouette wall)
    p.add_argument("--proto-dir", default=""); p.add_argument("--proto-init-dir", default="")
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--servo-radius", type=float, default=0.07)  # reach-then-servo: only trust the wrist once within this xy of the coarse goal
    p.add_argument("--ema", type=float, default=0.5)            # EMA on the wrist goal to kill per-step jitter
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    rng = np.random.RandomState(args.noise_seed)
    bank = None
    if args.coarse_src in ("dino", "sam", "triang", "depth"):
        print("building agentview DINOv2 prototype bank (honest coarse goal)...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    bank_w = None
    if args.wrist_id:
        print("building WRIST-view DINOv2 prototype bank (multi-view identity fusion)...", flush=True)
        bank_w = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                            args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "robot0_eye_in_hand", dev)
    if args.wrist_src == "rebind" and bank is None:
        print("building agentview DINOv2 prototype bank (for wrist re-bind)...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    gen = None
    if args.wrist_src in ("honest", "rebind") or args.coarse_src in ("sam", "triang", "depth"):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
        pps = 32 if args.coarse_src == "depth" else 16
        gen = SamAutomaticMaskGenerator(sam, points_per_side=pps, min_mask_region_area=25 if args.coarse_src == "depth" else 60)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"CLOSED-LOOP relocalize={args.relocalize} wrist_src={args.wrist_src} goal_noise={args.goal_noise} z_obj={args.z_obj}", flush=True)
    succ = []; nb = 0; reloc_used = []; close_loc = []; bind_ok = 0; coarse_err = []; R = args.res
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
        z_obj_task = args.z_obj
        if inits is not None and args.z_ref_init < len(inits):
            ze = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            ze.seed(777); ze.reset(); ze.set_init_state(inits[args.z_ref_init]); zsim = ze.env.sim
            zrb = resolve_bodies(zsim, [T])
            if zrb.get(T) is not None: z_obj_task = float(body_pos(zsim, zrb[T])[2])
            ze.close()
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables + [args.container + "_1"]); cb = rb[args.container + "_1"]
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            if rb.get(T) is None or cb is None: env.close(); succ.append(0); continue
            nb += 1
            obj_true = body_pos(sim, rb[T])
            if args.coarse_src == "depth":
                # HONEST foveated-id + DEPTH-median localization (~3cm, breaks silhouette wall) in a separate depths=True env
                from eval_e2e_depth import depth_localize
                dr = depth_localize(bf, inits[ti], T, bank, gen, dev, R, args.bind_res, z_obj_task, args.z_cont, args.container)
                if dr[0] is None: chosen = "__none__"; obj_w_coarse = np.array([0, 0, z_obj_task], np.float32)
                else: obj_w_coarse, _cw_depth, chosen = dr[0], dr[1], dr[2]
            elif args.coarse_src == "triang":
                # HONEST TWO-VIEW (no body_pos): agentview DINOv2 picks target identity+pixel; wrist mask whose ray comes
                # CLOSEST to the agentview ray (epipolar gate) is the same object -> triangulate the two rays -> ~1cm xy+z.
                proto = bank.get(nm(T)); HR = args.bind_res; sc = HR / R
                img_a = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
                hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
                pa = None; bestA = None
                for m in gen.generate(img_a):
                    seg = m["segmentation"]; a = int(seg.sum())
                    if a < 20 or a > 0.25 * R * R: continue
                    ys, xs = np.where(seg); ry, cx = ys.mean(), xs.mean()
                    hy, hx = int(ry * sc), int(cx * sc); s = 60
                    cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
                    if cr.size < 100: continue
                    score = float(dino_feat(cr, dev) @ proto) if proto is not None else 0.0
                    if bestA is None or score > bestA[0]: bestA = (score, R - 1 - ry, cx, seg)  # agentview pixel (row,col) + mask
                wr_img = np.asarray(sim.render(width=R, height=R, camera_name="robot0_eye_in_hand"))
                if args.wrist_flip: wr_img = wr_img[::-1].copy()
                if bestA is None: chosen = "__none__"; obj_w_coarse = np.array([0, 0, z_obj_task], np.float32)
                else:
                    if args.av_oracle:   # diag: true agentview pixel (isolate wrist-correspondence error from agentview id/pixel)
                        import robosuite.utils.camera_utils as cu
                        w2pa = cu.get_camera_transform_matrix(sim, "agentview", R, R)
                        pxa = cu.project_points_from_world_to_camera(obj_true[None], w2pa, R, R)[0]; pa = (pxa[0], pxa[1])
                    elif args.wrist_center == "dt":   # dt-peak (graspable interior pt) on the agentview mask too -> better epipolar ray
                        from scipy.ndimage import distance_transform_edt
                        dta = distance_transform_edt(bestA[3]); pk = np.unravel_index(int(np.argmax(dta)), dta.shape)
                        pa = (R - 1 - pk[0], pk[1])
                    else:
                        pa = (bestA[1], bestA[2])
                    oa, da = pixel_ray(sim, "agentview", pa[0], pa[1], R)
                    pw = None; bestRR = 1e9; bestID = -1e9; protoW = bank.get(nm(T)); sel_seg = None
                    bestFUSE = -1e9; protoWv = bank_w.get(nm(T)) if bank_w is not None else None
                    for m in gen.generate(wr_img):
                        seg = m["segmentation"]; a = int(seg.sum())
                        if a < 80 or a > 0.5 * R * R: continue          # size filter: a full object, not a part/the whole table
                        ys, xs = np.where(seg); rw, cw = ys.mean(), xs.mean()
                        ow, dw = pixel_ray(sim, "robot0_eye_in_hand", rw, cw, R)
                        rr = _ray_ray_dist(oa, da, ow, dw)
                        if args.wrist_id:
                            # MULTI-VIEW FUSION: wrist-view DINOv2 identity (independent signal) + agentview epipolar agreement
                            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                            cr = wr_img[max(0, y0 - 4):min(R, y1 + 4), max(0, x0 - 4):min(R, x1 + 4)]
                            if cr.size < 100: continue
                            idsc = float(dino_feat(cr, dev) @ protoWv) if protoWv is not None else 0.0
                            fuse = idsc - args.epi_w * rr
                            if fuse > bestFUSE: bestFUSE = fuse; pw = (rw, cw); sel_seg = seg
                        elif args.wrist_sel == "id_epi":
                            if rr > args.epi_thr: continue              # epipolar GATE, then rank by DINOv2 identity
                            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                            cr = wr_img[max(0, y0 - 4):min(R, y1 + 4), max(0, x0 - 4):min(R, x1 + 4)]
                            if cr.size < 100: continue
                            idsc = float(dino_feat(cr, dev) @ protoW) if protoW is not None else 0.0
                            if idsc > bestID: bestID = idsc; pw = (rw, cw); sel_seg = seg
                        else:
                            if rr < bestRR: bestRR = rr; pw = (rw, cw); sel_seg = seg  # min-ray-ray (baseline)
                    if sel_seg is not None and args.wrist_center != "centroid":
                        ys, xs = np.where(sel_seg)
                        if args.wrist_center == "bbox":
                            pw = ((ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0)
                        else:  # dt: distance-transform peak = deepest interior point (robust object-center proxy)
                            from scipy.ndimage import distance_transform_edt
                            dt = distance_transform_edt(sel_seg); pk = np.unravel_index(int(np.argmax(dt)), dt.shape)
                            pw = (float(pk[0]), float(pk[1]))
                    if pw is None: chosen = "__none__"; obj_w_coarse = np.array([0, 0, z_obj_task], np.float32)
                    else:
                        # WRIST-at-start single-view localization (near-overhead -> near-vertical ray -> xy ~z-insensitive):
                        # wrist ray ∩ plane at the object's calibrated height = ~0.7-1.2cm honest xy (beats triangulation,
                        # which is biased by silhouette-centroid non-correspondence across views). Agentview only IDs the target.
                        ow, dw = pixel_ray(sim, "robot0_eye_in_hand", pw[0], pw[1], R)
                        tw = (z_obj_task - ow[2]) / dw[2] if abs(dw[2]) > 0.1 else 0.0
                        pw3 = ow + tw * dw
                        obj_w_coarse = np.array([pw3[0], pw3[1], z_obj_task], np.float32)
                        import robosuite.utils.camera_utils as cu
                        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
                        pxt = cu.project_points_from_world_to_camera(obj_true[None], w2p, R, R)[0]
                        chosen = T if np.hypot(pa[0] - pxt[0], pa[1] - pxt[1]) < 22 else "__wrong__"
            elif args.coarse_src == "sam":
                # FULLY HONEST coarse goal (NO body_pos): SAM proposes regions on the agentview -> hi-res foveated crop ->
                # DINOv2 identity match -> base-contact pixel of the chosen mask -> ray-plane (~3cm). chosen=identity only.
                proto = bank.get(nm(T)); HR = args.bind_res; sc = HR / R
                img_a = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
                hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
                best = None
                for m in gen.generate(img_a):
                    seg = m["segmentation"]; a = int(seg.sum())
                    if a < 20 or a > 0.25 * R * R: continue
                    ys, xs = np.where(seg)
                    # MASK-BBOX crop at hi-res (object fills the crop -> better DINOv2 identity than a fixed centroid box)
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    pad = int(0.25 * max(y1 - y0, x1 - x0)) + 4
                    hy0, hy1 = int(max(0, y0 - pad) * sc), int(min(R, y1 + pad) * sc)
                    hx0, hx1 = int(max(0, x0 - pad) * sc), int(min(R, x1 + pad) * sc)
                    cr = hi[hy0:hy1, hx0:hx1]
                    if cr.size < 100: continue
                    f = dino_feat(cr, dev); score = float(f @ proto) if proto is not None else 0.0
                    if best is None or score > best[0]: best = (score, ys, xs)
                if best is None: chosen = "__none__"; obj_w_coarse = np.array([0, 0, z_obj_task], np.float32)
                else:
                    ys_b, xs_b = best[1], best[2]
                    import robosuite.utils.camera_utils as cu
                    w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R); inv = np.linalg.inv(w2p)
                    if args.loc_method == "centroid":
                        r_up = float(ys_b.mean()); c0 = float(xs_b.mean()); r0 = R - 1 - r_up; z_plane = z_obj_task  # object center @ its height
                    else:
                        base_up = int(ys_b.max()); c0 = int(xs_b[ys_b >= ys_b.max() - 2].mean()); r0 = R - 1 - base_up; z_plane = 0.0  # base contact @ table
                    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
                    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[min(max(r0,0),R-1), min(max(c0,0),R-1)]]), d1[None], inv)[0])
                    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[min(max(r0,0),R-1), min(max(c0,0),R-1)]]), d2[None], inv)[0])
                    ray = q2 - q1; tt = (z_plane - q1[2]) / ray[2]; pw = q1 + tt * ray
                    obj_w_coarse = np.array([pw[0], pw[1], z_obj_task], np.float32)
                    # identity check: is the chosen mask's base nearest to T's true projection? (for bind_ok accounting only)
                    pxt = cu.project_points_from_world_to_camera(obj_true[None], w2p, R, R)[0]
                    chosen = T if np.hypot(r0 - pxt[0], c0 - pxt[1]) < 22 else "__wrong__"
            elif args.coarse_src == "dino":
                # agentview foveated DINOv2 (body_pos crops => near-exact loc; identity test only)
                hi = sim.render(width=args.bind_res, height=args.bind_res, camera_name="agentview")
                up = np.asarray(hi)[::-1].copy(); proto = bank.get(nm(T)); feats = {}
                for o in graspables:
                    if rb.get(o) is None: continue
                    c = proto_crop(sim, up, args.bind_res, "agentview", rb[o], 60)
                    if c is not None: feats[o] = dino_feat(c, dev)
                chosen = max(feats, key=lambda o: float(feats[o] @ proto)) if (feats and proto is not None) else T
                obj_w_coarse = ray_plane(sim, rb[chosen], z_obj_task, R)
            else:
                # noise mode: true table-plane xy + injected coarse-loc error (stand-in for honest agentview loc; identity oracle)
                chosen = T
                noise = rng.randn(2) * args.goal_noise
                obj_w_coarse = np.array([obj_true[0] + noise[0], obj_true[1] + noise[1], z_obj_task], np.float32)
            bind_ok += int(chosen == T)
            if chosen == T: coarse_err.append(float(np.linalg.norm(obj_w_coarse[:2] - obj_true[:2])))  # diag (analysis only)
            cont_w = ray_plane(sim, cb, args.z_cont, R)
            z0 = obj_true[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            n_reloc = 0; min_close_err = 99.0; wrist_goal = None
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                obj_w = obj_w_coarse.copy()
                # CLOSED-LOOP reach-then-servo: reach on the coarse goal; only hand off to the wrist once the gripper is
                # CLOSE (within servo-radius of the coarse goal), where the object is centered & large in the wrist and
                # the table-plane ray is near-vertical (precise). EMA-smooth to kill per-step jitter.
                near = float(np.linalg.norm(ee[:2] - obj_w_coarse[:2])) < args.servo_radius
                # rebind engages by IDENTITY (not proximity) so it can steer the reach toward the right object from far
                gate = True if args.wrist_src == "rebind" else near
                do_reloc = (args.wrist_src == "oracle") or (chunk is None or ci >= args.replan)  # honest: only at replan (SAM is costly)
                if args.relocalize == "wrist" and not held and gate and do_reloc:
                    if args.wrist_src == "oracle":
                        pix, infov = wrist_proj(sim, body_pos(sim, rb[T]), R)   # oracle wrist pixel (architecture upper bound)
                    elif args.wrist_src == "rebind":
                        # RE-BIND on the wrist: run DINOv2 identity on each wrist mask (object is larger & top-down here ->
                        # identity easier than the 256px agentview) -> pick the mask that MATCHES the target prototype, not
                        # the one nearest a (wrong) coarse goal. Steers the reach to the right object even if coarse is 7cm off.
                        proto = bank.get(nm(T))
                        wr_img = np.asarray(sim.render(width=R, height=R, camera_name="robot0_eye_in_hand"))
                        if args.wrist_flip: wr_img = wr_img[::-1].copy()
                        pix = None; bests = -1e9
                        for m in gen.generate(wr_img):
                            seg = m["segmentation"]; a = int(seg.sum())
                            if a < 80 or a > 0.6 * R * R: continue
                            ys, xs = np.where(seg)
                            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                            cr = wr_img[max(0, y0 - 4):min(R, y1 + 4), max(0, x0 - 4):min(R, x1 + 4)]
                            if cr.size < 100: continue
                            sc_i = float(dino_feat(cr, dev) @ proto) if proto is not None else 0.0
                            if sc_i > bests and sc_i >= args.rebind_thr:
                                base = int(ys.max()); bc = int(xs[ys >= ys.max() - 2].mean())
                                bests = sc_i; pix = np.array([base, bc], np.float32)
                        infov = pix is not None
                    else:
                        # HONEST: SAM the wrist image; the coarse goal (carries identity from the agentview binder)
                        # projects to an expected wrist pixel -> pick the mask nearest it -> its base-contact pixel.
                        gpx, _ = wrist_proj(sim, obj_w_coarse, R)              # where our OWN coarse goal lands in the wrist
                        wr_img = np.asarray(sim.render(width=R, height=R, camera_name="robot0_eye_in_hand"))
                        if args.wrist_flip: wr_img = wr_img[::-1].copy()       # render bottom-origin -> match camera matrix
                        masks = gen.generate(wr_img); pix = None; bestd = 1e9
                        for m in masks:
                            seg = m["segmentation"]; a = int(seg.sum())
                            if a < 60 or a > 0.6 * R * R: continue
                            ys, xs = np.where(seg); cy, cx = ys.mean(), xs.mean()
                            d = float(np.hypot(cy - gpx[0], cx - gpx[1]))
                            if d < bestd and d < args.sel_radius:
                                base = int(ys.max()); bc = int(xs[ys >= ys.max() - 2].mean())
                                bestd = d; pix = np.array([base, bc], np.float32)
                        infov = pix is not None
                    if infov:
                        xy = wrist_ray_plane(sim, pix, 0.0, R)              # localize on the table plane via the wrist
                        meas = np.array([xy[0], xy[1], z_obj_task], np.float32)
                        wrist_goal = meas if wrist_goal is None else (args.ema * wrist_goal + (1 - args.ema) * meas)
                        n_reloc += 1
                        min_close_err = min(min_close_err, float(np.linalg.norm(meas[:2] - obj_true[:2])))
                if wrist_goal is not None and not held:
                    obj_w = wrist_goal.copy()
                goal_rel = ((cont_w if held else obj_w).astype(np.float32) - ee)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    theta = canon_angle(goal_rel); phi = -theta
                    wr_in = rot_image(wr, phi); g_in = rot_vec_xy(goal_rel, phi); pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                    img = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    vec = ((np.concatenate([g_in, pr, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        chh = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                    chunk = decanon_chunk(chh, theta); ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); reloc_used.append(n_reloc)
            if min_close_err < 99: close_loc.append(min_close_err)
            env.close()
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    cl = f"wrist_loc={np.mean(close_loc)*100:.2f}cm" if close_loc else "wrist_loc=n/a"
    ce = f"coarse_loc(bound)={np.mean(coarse_err)*100:.1f}cm" if coarse_err else "coarse_loc=n/a"
    print(f"\n=== CLOSED-LOOP coarse={args.coarse_src} wrist={args.wrist_src} reloc={args.relocalize} on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb}  {ce}  {cl}  reloc/ep={np.mean(reloc_used):.0f} ===  [swap pi0.5 17% | VLS 36.81%]", flush=True)
    print("CLWRIST_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
