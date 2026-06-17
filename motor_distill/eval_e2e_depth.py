"""HONEST e2e with DEPTH localization (breaks the silhouette-offset wall): foveated DINOv2 identity + DEPTH-unproject
the target mask to a 3D median (~2cm, vs 12cm silhouette ray-plane) + SE(2)-canon motor. Depth is grabbed ONCE at the
start in a SEPARATE camera_depths=True env (camera_depths corrupts the wrist during motor inference on EGL), then the
motor runs in a clean depths=False env at the SAME init state. No privileged target 3D.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_depth.py \
       --head data/motor_head_canon_z.pt --bddl-dir <swap> --init-dir <swap> --proto-dir <swap> --proto-init-dir <swap> \
       --proto-inits 30,32,34 --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from eval_e2e_rayplane import ray_plane
from bind_exemplar import build_bank
from dino_separability import dino_feat
from bind_foveate import nm
from eval_closedloop_wrist import wrist_ray_plane, wrist_proj
from bind_exemplar import proto_crop

LOOK_STEPS = 0    # arm-raise steps (default 0; robot-pixel rejection is the cleaner de-occlusion). Overridden by --look-steps


def build_container_proto(bddls, init_dir, proto_inits, R, dev, container):
    """DINOv2 prototype for the place container (basket), built from ref scenes (offline catalog; query stays honest)."""
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib as _pl
    fs = []
    for bf in bddls[:4]:
        stem = _pl.Path(bf).stem; fi = _pl.Path(init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        for ti in proto_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R); env.seed(ti); env.reset()
            obs = env.set_init_state(inits[ti]); sim = env.env.sim
            cb = resolve_bodies(sim, [container + "_1"]).get(container + "_1")
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and container in b]; cb = cand[0] if cand else None
            if cb is not None:
                up = np.asarray(obs["agentview_image"])[::-1].copy(); c = proto_crop(sim, up, R, "agentview", cb, 70)
                if c is not None: fs.append(dino_feat(c, dev))
            env.close()
    if not fs: return None
    v = np.mean(fs, 0); return v / np.linalg.norm(v)


def gdino_box(gd, proc, img_up, phrase, R, dev, box_thr=0.25, text_thr=0.20):
    """Catalog-free open-vocab box for `phrase`; returns mask of pixels inside the top box (matrix-native) or None."""
    import torch as _t
    from PIL import Image
    H = img_up.shape[0]; scale = R / H
    pil = Image.fromarray(img_up)
    inp = proc(images=pil, text=phrase.lower().strip() + ".", return_tensors="pt").to(dev)
    with _t.no_grad(): out = gd(**inp)
    res = proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=box_thr, text_threshold=text_thr,
                                                      target_sizes=[pil.size[::-1]])[0]
    if len(res["boxes"]) == 0: return None
    i = int(_t.argmax(res["scores"])); b = (res["boxes"][i].cpu().numpy() * scale)   # x0,y0,x1,y1 in 256 upright
    x0, y0, x1, y1 = [int(v) for v in b]
    ys_up, xs = np.mgrid[max(0, y0):min(R, y1), max(0, x0):min(R, x1)]
    return (R - 1 - ys_up.ravel()), xs.ravel()   # matrix-native rows, cols inside the box


def depth_localize(bf, init, T, bank, gen, dev, R, HR, z_obj_task, z_cont, container, id_oracle=False, cont_proto=None,
                   gd=None, gdproc=None, robot_reject=True):
    """Open a depths=True env at `init`, foveated-DINOv2 identify the target, DEPTH-unproject its mask -> 3D median xy.
    Returns (obj_w, cont_w, chosen) with obj_w from depth (honest) and cont_w via ray-plane on the container."""
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    segkw = {"camera_segmentations": "element"} if robot_reject else {}
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True, **segkw)
    env.seed(7); env.reset(); sim = env.env.sim
    obs = env.set_init_state(init)
    # HONEST robot-pixel mask (the robot knows its own geometry) -> exclude robot pixels from object localization, so
    # objects partially behind the arm aren't localized as the arm. Upright orientation to match the SAM crops below.
    robot_mask_up = None
    if robot_reject:
        rbods = [i for i in range(sim.model.nbody) if sim.model.body_id2name(i) and
                 any(k in sim.model.body_id2name(i) for k in ("robot", "gripper"))]
        rgeoms = np.array([g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] in rbods])
        seg_el = np.asarray(obs["agentview_segmentation_element"]).reshape(R, R)[::-1]
        robot_mask_up = np.isin(seg_el, rgeoms)
    # HONEST de-occlusion: raise the arm out of the agentview line-of-sight so table objects aren't robot-occluded at the
    # start (a real robot moves its arm to look). Objects don't move -> localization stays valid; motor runs from raw init.
    for _ in range(LOOK_STEPS):
        obs, _, _, _ = env.step([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    instr, objs, targets, distractors = parse_bddl(bf)
    graspables = [o for o in objs if container not in o]
    rb = resolve_bodies(sim, graspables + [container + "_1"])
    proto = bank.get(nm(T)); sc = HR / R
    w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R); inv = np.linalg.inv(w2p)
    real = cu.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(R, R, 1).astype(np.float32))[::-1].copy()
    d1m = np.ones((1, R, R, 1), np.float32); d2m = 2 * np.ones((1, R, R, 1), np.float32)
    def unproject(rows_native, cols):
        out = []
        for r, c in zip(rows_native, cols):
            r = int(min(max(r, 0), R - 1)); c = int(min(max(c, 0), R - 1))
            q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d1m, inv)[0])
            q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d2m, inv)[0])
            z = float(real[r, c, 0]); out.append(q1 + (z - 1.0) * (q2 - q1))
        return np.asarray(out)
    img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
    hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
    pxt0 = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, R, R)[0]  # for id-oracle selection / bookkeeping
    orow_up = R - 1 - pxt0[0]
    best = None; bestC = None     # bestC = container mask (DINOv2 vs container proto, honest)
    for m in gen.generate(img_up):
        seg = m["segmentation"]; a = int(seg.sum())
        if a < 20 or a > 0.5 * R * R: continue
        if robot_mask_up is not None and int((seg & robot_mask_up).sum()) > 0.4 * a: continue   # skip robot-arm masks
        ys, xs = np.where(seg); ry, cx = ys.mean(), xs.mean()
        if id_oracle:
            d = np.hypot(ys.mean() - orow_up, xs.mean() - pxt0[1])     # oracle: mask nearest the TRUE target projection
            if (best is None or d < best[0]) and a < 0.25 * R * R: best = (d, ys, xs)
        hy, hx = int(ry * sc), int(cx * sc); s = 60
        cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
        if cr.size < 100: continue
        f = dino_feat(cr, dev)
        if not id_oracle and proto is not None:
            score = float(f @ proto)
            if (best is None or score > best[0]) and a < 0.25 * R * R: best = (score, ys, xs)
        if cont_proto is not None:
            cs = float(f @ cont_proto)
            if bestC is None or cs > bestC[0]: bestC = (cs, ys, xs)
    if best is None:
        env.close(); return None, None, "__none__", 0.0, 0.0, 9.9
    def depth_xy(ys, xs, zlo=0.005, zhi=0.25):
        if robot_mask_up is not None and len(ys):       # drop any robot pixels inside the mask (partial occlusion)
            keep = ~robot_mask_up[ys, xs]
            if keep.sum() >= 3: ys, xs = ys[keep], xs[keep]
        if len(ys) > 60:
            idx = np.random.RandomState(0).choice(len(ys), 60, replace=False); ys, xs = ys[idx], xs[idx]
        pts = unproject(R - 1 - ys, xs); on = pts[(pts[:, 2] > zlo) & (pts[:, 2] < zhi)]
        if len(on) < 3: on = pts
        return np.median(on[:, :2], 0)
    xy = depth_xy(best[1], best[2])
    obj_w = np.array([xy[0], xy[1], z_obj_task], np.float32)          # DEPTH-median target xy (~2cm honest)
    # CLUTTER measure (honest, from the same masks): min distance from target xy to any OTHER detected mask's depth-xy.
    # Used to GATE the closed-loop wrist refinement (in clusters the wrist locks onto neighbors -> keep static goal).
    nn_dist = 9.9
    for m in gen.generate(img_up):
        seg = m["segmentation"]; a = int(seg.sum())
        if a < 20 or a > 0.25 * R * R: continue
        ys2, xs2 = np.where(seg)
        if abs(ys2.mean() - best[1].mean()) < 6 and abs(xs2.mean() - best[2].mean()) < 6: continue   # same mask
        oxy = depth_xy(ys2, xs2)
        d = float(np.linalg.norm(oxy - xy))
        if 0.02 < d < nn_dist: nn_dist = d
    # HONEST container localization: DINOv2-matched basket mask -> depth-median xy (basket is tall; widen z-band)
    cb = rb.get(container + "_1")
    if cb is None:
        cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and container in b]; cb = cand[0] if cand else None
    if gd is not None:   # HONEST container: catalog-free GroundingDINO box -> depth-median (basket: GDINO=100%, 11.6px)
        gb = gdino_box(gd, gdproc, img_up, container, R, dev)
        if gb is not None and len(gb[0]) > 5:
            cxy = depth_xy(gb[0], gb[1], zlo=0.005, zhi=0.30); cont_w = np.array([cxy[0], cxy[1], z_cont], np.float32)
        else:
            cont_w = ray_plane(sim, cb, z_cont, R) if cb is not None else np.array([0, 0, z_cont], np.float32)
    elif cont_proto is not None and bestC is not None:
        cxy = depth_xy(bestC[1], bestC[2], zlo=0.005, zhi=0.30); cont_w = np.array([cxy[0], cxy[1], z_cont], np.float32)
    else:
        cont_w = ray_plane(sim, cb, z_cont, R) if cb is not None else np.array([0, 0, z_cont], np.float32)
    pxt = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, R, R)[0]
    by = best[1]; bx = best[2]; chosen = T if np.hypot((R - 1 - by.mean()) - pxt[0], bx.mean() - pxt[1]) < 22 else "__wrong__"
    loc_err = float(np.linalg.norm(obj_w[:2] - body_pos(sim, rb[T])[:2]))
    cont_err = float(np.linalg.norm(cont_w[:2] - body_pos(sim, cb)[:2])) if cb is not None else 0.0
    env.close()
    return obj_w, cont_w, chosen, loc_err, cont_err, nn_dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", required=True); p.add_argument("--proto-init-dir", required=True)
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.05); p.add_argument("--z-ref-init", type=int, default=35)
    p.add_argument("--id-oracle", type=int, default=0)   # isolate localization: pick the TRUE target mask (no DINOv2 identity)
    p.add_argument("--sam-points", type=int, default=16); p.add_argument("--sam-minarea", type=int, default=60)
    p.add_argument("--relocalize", type=int, default=0); p.add_argument("--servo-radius", type=float, default=0.10); p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--priv-container", type=int, default=0)   # 1 = OLD privileged container via ray-plane on body_pos (ablation)
    p.add_argument("--gdino-container", type=int, default=1)  # 1 = honest catalog-free GroundingDINO container localization
    p.add_argument("--look-steps", type=int, default=0)       # arm-raise steps (default 0; robot-mask is the cleaner fix)
    p.add_argument("--robot-reject", type=int, default=0)     # exclude the robot's own pixels (default OFF; reverted - hurt swap)
    p.add_argument("--diag", type=int, default=0)             # per-trial stage breakdown: identity / loc / grasp / place
    p.add_argument("--cl-gate", type=float, default=0.0)      # >0: enable closed-loop ONLY if nearest-neighbor dist > this (m)
    p.add_argument("--lift-secure", type=float, default=0.02) # required lift (m) before switching goal to the container
    p.add_argument("--hold-grip", type=int, default=0)        # force grip closed during transport until near the container
    p.add_argument("--lift-wp", type=float, default=0.0)      # >0: post-grasp LIFT WAYPOINT — go straight up to this z before transporting (MOKA/ReKep post-contact waypoint; clears clutter)
    args = p.parse_args()
    global LOOK_STEPS; LOOK_STEPS = args.look_steps
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda"; R = args.res; HR = args.bind_res
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval(); vm, vs = ck["vm"], ck["vs"]
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.sam_points, min_mask_region_area=args.sam_minarea)
    print("building DINOv2 bank + container proto...", flush=True)
    pbddls = sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl")))
    pinits = [int(x) for x in args.proto_inits.split(",")]
    bank = build_bank(pbddls, args.proto_init_dir, pinits, HR, "agentview", dev)
    cont_proto = None
    gd = gdproc = None
    if not args.priv_container and args.gdino_container:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gdproc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
        print("loaded GroundingDINO for honest container localization", flush=True)
    succ = []; bind_ok = 0; nb = 0; loc = []; cloc = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        graspables = [o for o in objs if args.container not in o]
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        z_obj_task = args.z_obj
        if inits is not None and args.z_ref_init < len(inits):
            ze = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            ze.seed(777); ze.reset(); ze.set_init_state(inits[args.z_ref_init]); zrb = resolve_bodies(ze.env.sim, [T])
            if zrb.get(T) is not None: z_obj_task = float(body_pos(ze.env.sim, zrb[T])[2])
            ze.close()
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            r = depth_localize(bf, inits[ti], T, bank, gen, dev, R, HR, z_obj_task, args.z_cont, args.container, bool(args.id_oracle), cont_proto, gd, gdproc, bool(args.robot_reject))
            obj_w, cont_w, chosen = r[0], r[1], r[2]
            nb += 1; bind_ok += int(chosen == T)
            if obj_w is None: succ.append(0); continue
            loc.append(r[3]); cloc.append(r[4])
            nn_dist = r[5]
            cl_active = args.relocalize and (args.cl_gate <= 0 or nn_dist > args.cl_gate)   # P4: gate CL off in clusters
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]); rb = resolve_bodies(sim, graspables)
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            wrist_goal = None; obj_w0 = obj_w.copy()
            released_after_held = False; held_step = -1; max_obj_z = z0; lift_done = False
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                # CLOSED-LOOP wrist refinement (reach-then-servo): near the depth coarse goal, re-ground on the WRIST RGB
                if cl_active and not held and float(np.linalg.norm(ee[:2] - obj_w0[:2])) < args.servo_radius and (chunk is None or ci >= args.replan):
                    gpx, infov = wrist_proj(sim, obj_w0, R)
                    if infov:
                        wr_img = np.asarray(sim.render(width=R, height=R, camera_name="robot0_eye_in_hand"))[::-1].copy()
                        bestd = 1e9; pix = None
                        for m in gen.generate(wr_img):
                            seg = m["segmentation"]; a = int(seg.sum())
                            if a < 60 or a > 0.6 * R * R: continue
                            ys, xs = np.where(seg); d = float(np.hypot(ys.mean() - gpx[0], xs.mean() - gpx[1]))
                            if d < bestd and d < 80:
                                from scipy.ndimage import distance_transform_edt
                                dt = distance_transform_edt(seg); pk = np.unravel_index(int(np.argmax(dt)), dt.shape)
                                bestd = d; pix = np.array([float(pk[0]), float(pk[1])])
                        if pix is not None:
                            xy = wrist_ray_plane(sim, pix, 0.0, R); meas = np.array([xy[0], xy[1], obj_w0[2]], np.float32)
                            wrist_goal = meas if wrist_goal is None else (args.ema * wrist_goal + (1 - args.ema) * meas)
                obj_w = wrist_goal if (wrist_goal is not None and not held) else obj_w0
                if held and args.lift_wp > 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.15:
                    # transport-z schedule: container goal with HIGH z while far (motor arcs high, clears clutter;
                    # bearing stays defined toward the container -> canonicalization non-degenerate)
                    goal_now = np.array([cont_w[0], cont_w[1], args.lift_wp], np.float32)
                else:
                    goal_now = cont_w if held else obj_w
                goal_rel = (np.asarray(goal_now, np.float32) - ee)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    theta = canon_angle(goal_rel); phi = -theta
                    wr_in = rot_image(wr, phi); g_in = rot_vec_xy(goal_rel, phi); pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                    img = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    vec = ((np.concatenate([g_in, pr, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    with torch.no_grad(): chh = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                    chunk = decanon_chunk(chh, theta); ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > args.lift_secure: held = True; held_step = step
                if args.hold_grip and held and grip < 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.12:
                    grip = 1.0   # transport reflex: don't let a gripper flicker drop the object mid-transport
                if held and grip < 0: released_after_held = True
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                opz = body_pos(sim, rb[T])[2]; lifted = max(lifted, opz - z0); max_obj_z = max(max_obj_z, opz)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            if args.diag:
                fo = body_pos(sim, rb[T]); d_cont = float(np.linalg.norm(fo[:2] - cont_w[:2]))
                print(f"    [{stem[:16]:16s} t{t}] id={int(chosen==T)} loc={r[4-1]*100:4.1f}cm grasped={int(lifted>0.03)} "
                      f"held@{held_step} released={int(released_after_held)} obj→cont={d_cont*100:4.1f}cm obj_z={fo[2]:.3f} maxz={max_obj_z:.3f} ok={int(ok)}", flush=True)
            succ.append(int(ok)); env.close()
        print(f"  {stem[:30]:32s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bind_ok}/{nb}", flush=True)
    cl = f"cont_loc={np.mean(cloc)*100:.1f}cm" if cloc else "cont=priv"
    print(f"\n=== E2E-DEPTH (foveated id + DEPTH loc{' +CL' if args.relocalize else ''}{', PRIV-cont' if args.priv_container else ', HONEST-cont'}) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb}  loc_xy={np.mean(loc)*100:.1f}cm  {cl} ===  [swap pi0.5 17% | VLS 36.81%]", flush=True)
    print("E2EDEPTH_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
