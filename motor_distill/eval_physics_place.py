"""GROUNDED-PHYSICS PLACE (track 2 of the physics rethink). The reflex motor handles the GRASP (solved, ~1.0 with
hide). Once grasped, we STOP mimicking pi0.5 and instead compute the release from GROUNDED PHYSICS:
  - object pose while held = EE pose + grasp_offset  (offset measured ONCE at grasp -> honest, no continuous privilege)
  - container footprint + rim height from its geometry (here via geom AABB; at deploy via our 3D localizer)
  - drive the EE so the OBJECT's xy goes over the container center at z = rim + clearance, then OPEN -> gravity drops it IN
A grounded P-servo on delta-EE toward the physics-computed release target replaces the reflex place. This tests whether
grounded-physics targeting beats the reflex place (0.37). If yes, we distill the physics-planned place into the motor.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_physics_place.py \
  --head data/motor_head_selector.pt --bddl-dir <swap> --init-dir <swap> --container basket --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from collect_motor_data import surface_point
from node_localizer import localize as node_localize, flipped_depth as node_flipped


def ray_plane(sim, body, z_plane, R, cam="agentview"):
    """Depth-free localization: pixel ray (object center) x table plane z=z_plane. Validated ~0.1cm (eval_e2e_rayplane)."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0])
    ray = q2 - q1; t = (z_plane - q1[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0
    pt = q1 + t * ray; return np.array([pt[0], pt[1], z_plane], np.float32)


def depth_localize(sim, px, depth_real, R, win=7, cam="agentview", mode="object"):
    """Per-object 3D from a depth window around pixel px (depth_real = linearized, row-aligned metric depth).
    mode='object': upper-half-z median (object top surface, ~2cm). mode='container': CENTROID xy over the whole region
    (unbiased opening center, not rim-ring-biased) + high-percentile z (rim top) -> for placing INTO a container."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    r0, c0 = int(min(max(px[0], 0), R - 1)), int(min(max(px[1], 0), R - 1))
    dm = depth_real[None]; pts = []
    for r in range(max(0, r0 - win), min(R, r0 + win + 1), 2):
        for c in range(max(0, c0 - win), min(R, c0 + win + 1), 2):
            pts.append(np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), dm, inv)[0]))
    if not pts: return None
    pts = np.array(pts)
    if mode == "container":
        xy = np.median(pts[:, :2], axis=0); z = np.percentile(pts[:, 2], 80)   # opening centroid xy + rim z
        return np.array([xy[0], xy[1], z], np.float32)
    zmed = np.median(pts[:, 2]); keep = pts[pts[:, 2] >= zmed]                  # object: upper-half-z surface
    return np.median(keep, axis=0).astype(np.float32) if len(keep) else np.median(pts, axis=0).astype(np.float32)


def obj_pixel(sim, body, R, cam="agentview"):
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    return px


def body_aabb_top(sim, body):
    """Top z and xy half-extent of a body from its geom sizes (container rim estimate)."""
    bid = sim.model.body_name2id(body) if isinstance(body, str) else body
    zs = []; halfxy = 0.0
    for g in range(sim.model.ngeom):
        if sim.model.geom_bodyid[g] != bid: continue
        gp = sim.data.geom_xpos[g]; sz = sim.model.geom_size[g]
        zs.append(gp[2] + sz[2]); halfxy = max(halfxy, float(max(sz[0], sz[1])))
    top = max(zs) if zs else float(body_pos(sim, bid)[2] + 0.05)
    return top, halfxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_selector.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--flow", default="")   # path to FlowMotor ckpt -> use flow-matching head (learned-place fix) instead of L1
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--canon-mode", default="perstep_obj"); p.add_argument("--hide", type=int, default=1)
    p.add_argument("--clear", type=float, default=0.10)   # release height above rim
    p.add_argument("--gain", type=float, default=8.0); p.add_argument("--tol", type=float, default=0.025)
    p.add_argument("--reflex", type=int, default=0)       # 0=physics place, 1=pure reflex (baseline A/B)
    p.add_argument("--binder", default="oracle", choices=["oracle", "dino", "sam"])   # sam = SAM proposes pixels (no body_pos) -> FULLY honest attention
    p.add_argument("--sam-ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--loc", default="oracle", choices=["oracle", "rayplane", "depthflip", "nodeloc"])   # depthflip/nodeloc=honest depth localizer (nodeloc=general masked-depth, validated ~1.7cm)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.04)
    p.add_argument("--rim-h", type=float, default=0.075)   # honest container rim height above its table-plane z (basket ~7.5cm)
    p.add_argument("--z-cont-center", type=float, default=0.035)   # basket center height (class const) for ray-plane container loc (avoids depth parallax)
    p.add_argument("--obj-oracle-z", type=int, default=0)  # DIAGNOSTIC: honest xy but oracle object z (isolates depth top-surface vs body-center z-shift on grasp)
    p.add_argument("--obj-z-off", type=float, default=0.0)  # lower the object goal z toward the grasp point (depth gives TOP surface)
    p.add_argument("--grasp-lift", type=float, default=0.02)  # EE-rise required before physics takeover (secure grasp first; reflex grasps 1.0 so 0.02 takes over too early)
    p.add_argument("--obj-aware-release", type=int, default=0)  # OBJECT-aware release: target the OBJECT (ee+grasp_off) at rim+clear, not the EE -> fixes bigger-object place
    p.add_argument("--proto-dir", default=""); p.add_argument("--proto-init-dir", default="")
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    args = p.parse_args()
    HONEST = (args.loc in ("rayplane", "depthflip", "nodeloc")); DEPTHCAM = (args.loc in ("depthflip", "nodeloc"))
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    fnet = None; cmf = csf = None; fsteps = 10
    if args.flow:
        from train_flowmotor import FlowMotor
        fck = torch.load(args.flow, map_location=dev, weights_only=False)
        fnet = FlowMotor(prop_dim=fck["prop_dim"]).to(dev); fnet.load_state_dict(fck["state"]); fnet.eval()
        vm, vs = fck["vm"], fck["vs"]; cmf = fck["cm"]; csf = fck["cs"]; fsteps = int(fck.get("infer_steps", 10))
        print("FLOW head loaded steps=" + str(fsteps), flush=True)
    bank = None; sam_gen = None
    if args.binder in ("dino", "sam"):
        from bind_exemplar import build_bank
        print("building DINOv2 prototype bank...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    if args.binder == "sam":
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt).to(dev).eval()
        sam_gen = SamAutomaticMaskGenerator(sam, points_per_side=24, min_mask_region_area=40)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"PHYSICS-PLACE reflex={args.reflex} clear={args.clear} on {pathlib.Path(args.bddl_dir).name}", flush=True)
    succ = []; grasp = []; placed_phys = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem; graspables = [o for o in objs if args.container not in o]
        import re as _re
        m = _re.match(r"pick_up_the_(.+)$", stem); T = targets[0]
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for s in ("between", "next", "on", "from", "in", "and"):
                if s in toks: cut = min(cut, toks.index(s))
            tc = "_".join(toks[:cut]); cand = next((o for o in graspables if tc and tc in o), None)
            if cand: T = cand
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=DEPTHCAM)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables + [args.container + "_1"]); cb = rb.get(args.container + "_1")
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            if rb.get(T) is None or cb is None: env.close(); succ.append(0); grasp.append(0); continue
            # ---- BIND target identity ON THE FULL SCENE (must be BEFORE hide!) ----
            obj_px = None   # SAM-proposed pixel (fully-honest attention, NO body_pos); None -> oracle obj_pixel(body_pos)
            if args.binder == "oracle":
                chosen = T
            elif args.binder == "dino":
                from bind_exemplar import proto_crop
                from dino_separability import dino_feat
                from bind_foveate import nm
                hi = sim.render(width=args.bind_res, height=args.bind_res, camera_name="agentview")
                up = np.asarray(hi)[::-1].copy(); proto = bank.get(nm(T)); feats = {}
                for o in graspables:
                    if rb.get(o) is None: continue
                    c = proto_crop(sim, up, args.bind_res, "agentview", rb[o], 60)
                    if c is not None: feats[o] = dino_feat(c, dev)
                chosen = max(feats, key=lambda o: float(feats[o] @ proto)) if (feats and proto is not None) else T
            else:   # sam: SAM proposes regions (NO body_pos) -> foveated-DINOv2 match to target proto -> region pixel
                from dino_separability import dino_feat
                from bind_foveate import nm
                R = args.res; HR = args.bind_res; sc = HR / R
                img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
                hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
                proto = bank.get(nm(T)); best = None
                for m in sam_gen.generate(img_up):
                    seg = m["segmentation"]; a = int(seg.sum())
                    if a < 20 or a > 0.25 * R * R: continue
                    ys, xs = np.where(seg); ry, cx = float(ys.mean()), float(xs.mean())
                    hy, hx = int(ry * sc), int(cx * sc); s = 60
                    cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
                    if cr.size < 100: continue
                    f = dino_feat(cr, dev); score = float(f @ proto) if proto is not None else 0.0
                    if best is None or score > best[0]: best = (score, ry, cx)
                chosen = T
                if best is not None:
                    obj_px = np.array([R - 1 - best[1], best[2]], np.float32)   # upright-row -> projection frame
            if obj_px is not None:   # SAM scoring: pixel distance to the true target pixel
                _tpx = obj_pixel(sim, rb[T], args.res); bind_ok += int(np.hypot(obj_px[0] - _tpx[0], obj_px[1] - _tpx[1]) < 18)
            else:
                bind_ok += int(chosen == T)
            nb += 1
            # ---- HIDE all non-chosen graspables (binder->HIDE->motor), AFTER binding ----
            if args.hide:
                for o in graspables:
                    if o == chosen or rb.get(o) is None: continue
                    bid = sim.model.body_name2id(rb[o])
                    for g in range(sim.model.ngeom):
                        if sim.model.geom_bodyid[g] == bid: sim.model.geom_rgba[g, 3] = 0.0
            # ---- LOCALIZE (oracle body_pos vs OUR honest ray-plane) ----
            if args.loc == "depthflip":
                import robosuite.utils.camera_utils as _cu
                dm = _cu.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]))[::-1].copy()   # row-flip aligns depth to projection (~2cm)
                obj_w = depth_localize(sim, (obj_px if obj_px is not None else obj_pixel(sim, rb[chosen], args.res)), dm, args.res, mode="object")
                cont_w = ray_plane(sim, cb, args.z_cont_center, args.res)   # ray-plane at container-center height: avoids basket-rim DEPTH PARALLAX (~0.2cm vs 7cm)
                if obj_w is None: obj_w = body_pos(sim, rb[chosen]).astype(np.float32)
                if args.obj_oracle_z: obj_w[2] = float(body_pos(sim, rb[chosen])[2])   # DIAGNOSTIC: honest xy, oracle z
                obj_w[2] += args.obj_z_off                                              # lower goal toward grasp point
                rim_top = args.z_cont_center + args.rim_h   # release above the rim
            elif args.loc == "nodeloc":
                dm = node_flipped(sim, obs["agentview_depth"])                          # validated general masked-depth localizer (~1.7cm)
                obj_w = node_localize(sim, (obj_px if obj_px is not None else obj_pixel(sim, rb[chosen], args.res)), dm, args.res, win=12, z_mode="surface")
                cont_w = ray_plane(sim, cb, args.z_cont_center, args.res)
                if obj_w is None: obj_w = body_pos(sim, rb[chosen]).astype(np.float32)
                if args.obj_oracle_z: obj_w[2] = float(body_pos(sim, rb[chosen])[2])
                obj_w[2] += args.obj_z_off
                rim_top = args.z_cont_center + args.rim_h
            elif args.loc == "rayplane":
                obj_w = ray_plane(sim, rb[chosen], args.z_obj, args.res); cont_w = ray_plane(sim, cb, args.z_cont, args.res)
                rim_top = args.z_cont + args.rim_h
            else:
                obj_w = body_pos(sim, rb[chosen]).astype(np.float32); cont_w = body_pos(sim, cb).astype(np.float32)
                rim_top, _ = body_aabb_top(sim, sim.model.body_name2id(cb))
            ee_z0 = float(np.asarray(obs["robot0_eef_pos"], np.float32)[2]); ee_zmin = ee_z0
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; close_run = 0; held = False; grasp_off = None; released = 0
            chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                # ---- GRASP phase: reflex motor servos to the object ----
                if not held or args.reflex:
                    obj_rel = obj_w - ee; cont_rel = cont_w - ee
                    if chunk is None or ci >= args.replan:
                        wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                        wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                        prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                               np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                        phi = -canon_angle(obj_rel) if args.canon_mode == "perstep_obj" else 0.0; theta = -phi
                        if phi:
                            wr_in = rot_image(wr, phi); o_in = rot_vec_xy(obj_rel, phi); c_in = rot_vec_xy(cont_rel, phi)
                            pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                        else:
                            wr_in = wr; o_in = obj_rel; c_in = cont_rel; pr = prop
                        img = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                        vec = ((np.concatenate([o_in, c_in, pr]).astype(np.float32) - vm) / vs).astype(np.float32)
                        with torch.no_grad():
                            if fnet is not None:
                                chs = fnet.sample(img, torch.tensor(vec)[None].to(dev), steps=fsteps).cpu().numpy()[0].reshape(-1)
                                chh = (chs * csf + cmf).reshape(fnet.chunk, fnet.act)
                            else:
                                chh = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                        chunk = decanon_chunk(chh, theta) if phi else chh; ci = 0
                    a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_run = close_run + 1 if grip > 0 else 0
                    act = a[:7].copy(); act[6] = grip
                else:
                    # ---- GROUNDED-PHYSICS PLACE: drive object over container, release above rim ----
                    tgt_z = rim_top + args.clear   # rim_top precomputed (geom AABB oracle, or z_cont+rim_h honest)
                    if args.obj_aware_release: tgt_z = tgt_z - float(grasp_off[2])   # place the OBJECT (ee+grasp_off), not the EE, at rim+clear
                    des_ee = np.array([cont_w[0] - grasp_off[0], cont_w[1] - grasp_off[1], tgt_z], np.float32)
                    over = np.linalg.norm((ee + grasp_off)[:2] - cont_w[:2]) < args.tol and abs(ee[2] - tgt_z) < args.tol
                    if over or released:
                        released += 1
                        act = np.zeros(7, np.float32); act[6] = -1.0     # open to drop
                    else:
                        d = np.clip((des_ee - ee) * args.gain, -1.0, 1.0)
                        act = np.zeros(7, np.float32); act[:3] = d; act[6] = 1.0   # hold closed, servo to release pose
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, float(body_pos(sim, rb[T])[2] - z0))   # privileged (metric only)
                ee_now = np.asarray(obs["robot0_eef_pos"], np.float32); ee_zmin = min(ee_zmin, float(ee_now[2]))
                grasped = (close_run > 8 and (float(ee_now[2]) - ee_zmin > args.grasp_lift)) if HONEST else (close_run > 8 and lifted > 0.02)
                if (not held) and grasped:
                    held = True
                    grasp_off = (obj_w - ee_now) if HONEST else (body_pos(sim, rb[T]).astype(np.float32) - ee_now)
                if released > 25: break
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); grasp.append(int(lifted > 0.03)); placed_phys.append(int(held and not args.reflex)); env.close()
        print(f"  {stem[:30]:32s} succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp={np.mean(grasp):.2f}", flush=True)
    print(f"\n=== PHYSICS-PLACE [reflex={args.reflex}] on {pathlib.Path(args.bddl_dir).name}: "
          f"succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp_rate={np.mean(grasp):.3f}  bind={bind_ok}/{nb} ===  [reflex-place baseline ~0.37]", flush=True)
    print("PHYSPLACE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
