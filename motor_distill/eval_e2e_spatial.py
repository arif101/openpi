"""SPATIAL suite: identical black bowls referred by RELATION ("the bowl between the plate and the ramekin"). Appearance
can't disambiguate -> build an honest 3D SCENE GRAPH (SAM masks -> DINOv2 class -> depth-median xy for ALL objects) and
ground the relation GEOMETRICALLY: between(A,B)=min d(b,A)+d(b,B); next_to(A)=min d(b,A); on(A)=xy within A + z above;
table_center=min dist to workspace center. Then the validated canon motor picks the chosen bowl -> places on the plate.
Relation parsed rule-based from the instruction (the task stem). Catalog = per-CLASS DINOv2 protos from ref scenes.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv/bin/python motor_distill/eval_e2e_spatial.py \
       --head data/motor_head_canon_z.pt --bddl-dir <spatial_x> --init-dir <spatial_x> --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from dino_separability import dino_feat


def cls_of(name):  # "akita_black_bowl_1" -> "akita_black_bowl"
    return re.sub(r"_\d+$", "", name)


def target_cls_from_stem(stem):
    """INSTRUCTION-derived pick target (general; no BDDL goal peeking): 'pick_up_the_<X>_<relation>...' -> X."""
    m = re.match(r"pick_up_the_(.+)$", stem)
    if not m: return None
    toks = m.group(1).split("_")
    cut = len(toks)
    for stop in ("between", "next", "on", "from", "in", "and"):
        if stop in toks: cut = min(cut, toks.index(stop))
    return "_".join(toks[:cut]) or None


def parse_relation(stem):
    """Rule-based relation from the task stem. Returns (kind, anchors)."""
    m = re.search(r"between_the_(\w+?)_and_the_(\w+?)_and", stem)
    if m: return ("between", [m.group(1), m.group(2)])
    m = re.search(r"next_to_the_(\w+?)_and", stem)
    if m: return ("next_to", [m.group(1)])
    m = re.search(r"on_the_(\w+?)_and", stem)
    if m: return ("on", [m.group(1)])
    if "from_table_center" in stem: return ("center", [])
    m = re.search(r"in_the_top_drawer", stem)
    if m: return ("drawer", [])
    return ("any", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--proto-inits", default="30,32,34")
    p.add_argument("--bind-res", type=int, default=1024); p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=380); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.02); p.add_argument("--z-place", type=float, default=0.06)
    p.add_argument("--sam-points", type=int, default=32); p.add_argument("--sam-minarea", type=int, default=25)
    p.add_argument("--lift-secure", type=float, default=0.06); p.add_argument("--hold-grip", type=int, default=1); p.add_argument("--lift-wp", type=float, default=0.28)
    p.add_argument("--diag", type=int, default=1)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import robosuite.utils.camera_utils as cu
    dev = "cuda"; R = args.res; HR = args.bind_res; sc = HR / R
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval(); vm, vs = ck["vm"], ck["vs"]
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.sam_points, min_mask_region_area=args.sam_minarea)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]

    # ---- per-CLASS prototype bank from ref scenes (exemplar catalog: crop each named object in ref inits) ----
    print("building per-class DINOv2 bank for spatial suite...", flush=True)
    acc = {}
    pinits = [int(x) for x in args.proto_inits.split(",")]
    for bf in bddls[:4]:
        instr, objs, targets, distractors = parse_bddl(bf)
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        for ti in pinits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R); env.seed(ti); env.reset()
            obs = env.set_init_state(inits[ti]); sim = env.env.sim
            hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            names = list(dict.fromkeys(objs))
            rb = resolve_bodies(sim, names)
            for o in names:
                if rb.get(o) is None: continue
                px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
                hy, hx = int((R - 1 - px[0]) * sc), int(px[1] * sc); s = 70
                cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
                if cr.size < 100: continue
                acc.setdefault(cls_of(o), []).append(dino_feat(cr, dev))
            env.close()
    bank = {k: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for k, v in acc.items() if v}
    print(f"classes: {sorted(bank)}", flush=True)

    succ = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem
        rel, anchors = parse_relation(stem)
        # INSTRUCTION-derived pick target (general; BDDL targets[0] can be the PLATE on spatial goals)
        tcls = target_cls_from_stem(stem) or cls_of(targets[0])
        bowl_cls = next((k for k in bank if k == tcls), None) or \
                   next((k for k in bank if tcls.endswith(k) or k.endswith(tcls)), None) or \
                   next((k for k in bank if "bowl" in k and "bowl" in tcls), tcls)
        # GT instance of the SAME class (for diagnostics + lift tracking): prefer a target entry, else any obj
        T = next((o for o in targets if cls_of(o).endswith(bowl_cls) or bowl_cls.endswith(cls_of(o))), None) or \
            next((o for o in objs if cls_of(o).endswith(bowl_cls) or bowl_cls.endswith(cls_of(o))), targets[0])
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            # ---- scene graph from a depths=True env (closed before the motor env) ----
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
            env.seed(7); env.reset(); sim = env.env.sim; obs = env.set_init_state(inits[ti])
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R); inv = np.linalg.inv(w2p)
            real = cu.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(R, R, 1).astype(np.float32))[::-1].copy()
            d1m = np.ones((1, R, R, 1), np.float32); d2m = 2 * np.ones((1, R, R, 1), np.float32)
            def unproj(rr, cc):
                rr = int(min(max(rr, 0), R - 1)); cc = int(min(max(cc, 0), R - 1))
                q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d1m, inv)[0])
                q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d2m, inv)[0])
                z = float(real[rr, cc, 0]); return q1 + (z - 1.0) * (q2 - q1)
            img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
            hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
            # GENERAL support-surface estimation from depth (no scene constants): unproject a uniform pixel grid; the
            # dominant z mode = the surface (floor scenes z~0, kitchen tables z~0.9, any deployment).
            gz = [unproj(r, c)[2] for r in range(8, R, 24) for c in range(8, R, 24)]
            hzs, hedges = np.histogram(gz, bins=60)
            z_s = float((hedges[np.argmax(hzs)] + hedges[np.argmax(hzs) + 1]) / 2)
            det = []   # (class, score, xy3d, z3d)
            for m in gen.generate(img_up):
                seg = m["segmentation"]; a = int(seg.sum())
                if a < 20 or a > 0.3 * R * R: continue
                ys, xs = np.where(seg)
                # MASK-TIGHT hi-res crop (object fills the crop -> better fine-grained identity than fixed window)
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                pad = int(0.25 * max(y1 - y0, x1 - x0)) + 3
                cr = hi[int(max(0, y0 - pad) * sc):int(min(R, y1 + pad) * sc), int(max(0, x0 - pad) * sc):int(min(R, x1 + pad) * sc)]
                if cr.size < 100: continue
                f = dino_feat(cr, dev)
                scores = {k: float(f @ v) for k, v in bank.items()}
                k = max(scores, key=scores.get)
                if len(ys) > 50:
                    idx = np.random.RandomState(0).choice(len(ys), 50, replace=False); ys, xs = ys[idx], xs[idx]
                pts = np.asarray([unproj(R - 1 - y, x) for y, x in zip(ys, xs)])
                on = pts[(pts[:, 2] > z_s + 0.005) & (pts[:, 2] < z_s + 0.40)]   # relative to ESTIMATED surface
                if len(on) < 3: continue
                # METRIC size gate (general): physical extent from depth; manipulable tabletop objects are 3-20cm wide.
                ext = float(max(on[:, 0].max() - on[:, 0].min(), on[:, 1].max() - on[:, 1].min()))
                if ext < 0.03 or ext > 0.22: continue
                # depth-coherence gate: a real object's points cluster in z (cabinet-front patches smear vertically)
                if float(on[:, 2].max() - on[:, 2].min()) > 0.18: continue
                det.append((k, scores[k], np.median(on[:, :2], 0), float(np.median(on[:, 2]))))
            bowls = [d for d in det if d[0] == bowl_cls]
            # dedup bowls closer than 4cm (fragmented masks)
            uniq = []
            for b in sorted(bowls, key=lambda x: -x[1]):
                if all(np.linalg.norm(b[2] - u[2]) > 0.04 for u in uniq): uniq.append(b)
            anchor_xy = {}
            for aname in anchors + ["plate"]:
                cands = [d for d in det if aname in d[0]]
                if cands: anchor_xy[aname] = max(cands, key=lambda x: x[1])[2]
            # ---- geometric relation grounding (xy + each candidate's MEASURED 3D z) ----
            chosen = None
            if uniq:
                if rel == "between" and all(a in anchor_xy for a in anchors):
                    A, B = anchor_xy[anchors[0]], anchor_xy[anchors[1]]
                    chosen = min(uniq, key=lambda b: np.linalg.norm(b[2] - A) + np.linalg.norm(b[2] - B))
                elif rel == "next_to" and anchors[0] in anchor_xy:
                    chosen = min(uniq, key=lambda b: np.linalg.norm(b[2] - anchor_xy[anchors[0]]))
                elif rel == "on" and anchors and anchors[0] in anchor_xy:
                    # "on the X": prefer ELEVATED bowls near X (z above surface by > 6cm), else nearest to X
                    elev = [b for b in uniq if b[3] > z_s + 0.06]
                    pool = elev if elev else uniq
                    chosen = min(pool, key=lambda b: np.linalg.norm(b[2] - anchor_xy[anchors[0]]))
                elif rel == "center":
                    cxy = np.mean([d[2] for d in det], 0)
                    chosen = min(uniq, key=lambda b: np.linalg.norm(b[2] - cxy))
                else:
                    chosen = uniq[0]
            chosen_xy = None if chosen is None else chosen[2]; chosen_z = None if chosen is None else chosen[3]
            rbT = resolve_bodies(sim, [T]); gt = body_pos(sim, rbT[T]) if rbT.get(T) is not None else None
            nb += 1
            ok_bind = chosen_xy is not None and gt is not None and np.linalg.norm(chosen_xy - gt[:2]) < 0.06
            bind_ok += int(ok_bind)
            if args.diag:
                dd = {}
                for k, s, xy, zz in det: dd.setdefault(k, []).append(f"({xy[0]:.2f},{xy[1]:.2f},z{zz:.2f})")
                print(f"      T={T} tcls={tcls} bowl_cls={bowl_cls}", flush=True)
                print(f"      z_surface={z_s:.3f}  DET: { {k: v for k, v in dd.items()} }", flush=True)
                print(f"      bowls(dedup)={[f'({b[2][0]:.2f},{b[2][1]:.2f},z{b[3]:.2f})' for b in uniq]}  anchors={ {a: f'({v[0]:.2f},{v[1]:.2f})' for a, v in anchor_xy.items()} }", flush=True)
                print(f"      chosen={None if chosen_xy is None else f'({chosen_xy[0]:.2f},{chosen_xy[1]:.2f})'}  GT=({gt[0]:.2f},{gt[1]:.2f},z={gt[2]:.2f})" if gt is not None else "      GT none", flush=True)
            plate_xy = anchor_xy.get("plate")
            env.close()
            if chosen_xy is None or plate_xy is None: succ.append(0); continue
            obj_w = np.array([chosen_xy[0], chosen_xy[1], chosen_z], np.float32)                      # MEASURED 3D z (general)
            cont_w = np.array([plate_xy[0], plate_xy[1], z_s + args.z_place], np.float32)             # place height relative to estimated surface
            # ---- motor (clean env) ----
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]); rbT = resolve_bodies(sim, [T])
            z0 = body_pos(sim, rbT[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                if held and args.lift_wp > 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.15:
                    goal_now = np.array([cont_w[0], cont_w[1], args.lift_wp], np.float32)
                else:
                    goal_now = cont_w if held else obj_w
                goal_rel = (goal_now - ee)
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
                if (not held) and close_cnt > 8 and lifted > args.lift_secure: held = True
                if args.hold_grip and held and grip < 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.12: grip = 1.0
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rbT[T])[2] - z0)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            if args.diag:
                print(f"    [{stem[:34]:36s} t{t}] rel={rel:8s} bind={int(ok_bind)} grasped={int(lifted>0.03)} ok={int(ok)}", flush=True)
            succ.append(int(ok)); env.close()
        print(f"  {stem[:40]:42s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind={bind_ok}/{nb}", flush=True)
    print(f"\n=== E2E-SPATIAL (scene-graph relation grounding + canon motor) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  bind_ok={bind_ok}/{nb} ===  [VLS spatial: task 54 / pos 42]", flush=True)
    print("E2ESPATIAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
