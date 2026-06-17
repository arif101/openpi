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
from train_wristcam_motor import WristMotor, CC_VOCAB, cont_class_from_name
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
    top = max(zs) if zs else float(sim.data.body_xpos[bid][2] + 0.05)   # bid already an id (fixtures w/ geoms in child bodies)
    return top, halfxy


def scene_bodies(bf):
    """Object + fixture instance names from the BDDL (the scene the agent perceives) — NOT the (:goal) predicate."""
    import re as _r, pathlib as _p
    txt = _p.Path(bf).read_text(); names = []
    for blk in (_r.search(r"\(:objects\s+(.+?)\)\s*\(:", txt, _r.S), _r.search(r"\(:fixtures\s+(.+?)\)\s*\(:", txt, _r.S)):
        if blk: names += _r.findall(r"(\w+_\d+)", blk.group(1))
    return list(dict.fromkeys(names))


def resolve_noun(noun, candidates):
    """Open-vocab noun -> scene body by token overlap (language-driven; no body_pos / no goal predicate)."""
    import re as _r
    nt = set(_r.sub(r"[^a-z ]", " ", (noun or "").lower()).split())
    best, bs = None, 0
    for b in candidates:
        bt = set(_r.sub(r"_\d+$", "", b).split("_"))
        ov = len(nt & bt)
        if ov > bs: bs, best = ov, b
    return best if bs > 0 else None


def _body_tokens(b):
    import re as _r
    return set(_r.sub(r"_\d+$", "", b).split("_"))


def select_instance(sim, obj_noun, relation, names):
    """RELATIONAL which-instance (de-hardcode oracle which-bowl): among same-noun candidate instances,
    pick the one satisfying `relation` vs the reference objects' positions — GEOMETRY, not obj_of_interest.
    (Positions via body_pos here = residual; the de-hardcode is the SELECTION MECHANISM = relation+geometry.)"""
    from language_planner import parse_relation, resolve_relation
    nt = set((obj_noun or "").lower().split())
    cands = [n for n in names if len(nt & _body_tokens(n)) >= max(1, len(nt) - 1)]
    if len(cands) <= 1:
        return cands[0] if cands else None
    _, refnouns = parse_relation(relation)
    refkeys = {}
    for rn in refnouns:
        rt = set(rn.lower().split())
        best = max(names, key=lambda n: len(rt & _body_tokens(n)), default=None)
        if best and len(rt & _body_tokens(best)) > 0:
            refkeys[rn] = best
    rb = resolve_bodies(sim, cands + list(refkeys.values()))
    cand_pos = [(k, body_pos(sim, rb[k])) for k in cands if rb.get(k) is not None]
    ref_pos = {rn: body_pos(sim, rb[refkeys[rn]]) for rn in refkeys if rb.get(refkeys[rn]) is not None}
    if len(cand_pos) <= 1:
        return cand_pos[0][0] if cand_pos else None
    return resolve_relation(relation, cand_pos, ref_pos)


def bind_target(sim, obs, args, T, graspables, rb, dev, bank, sam_gen):
    """Identity binding: returns (chosen_body_name, obj_px or None). obj_px=SAM-proposed pixel (honest attention)."""
    obj_px = None
    if args.binder == "oracle":
        return T, None
    if args.binder == "learned":
        # LEARNED grounding head (trained on LIBERO sim auto-labels): frozen DINOv2 dense feats + object proto -> heatmap.
        from bind_foveate import nm
        from diag_dinodense import dino_dense
        from train_ground_head import GroundHead
        global _GH
        if "_GH" not in globals() or _GH is None:
            gk = torch.load(args.ground_head, map_location=dev, weights_only=False)
            gh = GroundHead(gk["C"]).to(dev); gh.load_state_dict(gk["state"]); gh.eval()
            globals()["_GH"] = (gh, gk)
        gh, gk = globals()["_GH"]
        R = args.res; gres = gk["res"]
        proto = gk["protos"].get(nm(T))
        if proto is None: return T, None
        hi = np.asarray(sim.render(width=gres, height=gres, camera_name="agentview"))[::-1].copy()
        fg, g = dino_dense(hi, dev, gres)
        with torch.no_grad():
            lo = gh(torch.tensor(fg[None]).to(dev), torch.tensor(proto[None]).to(dev))[0]
            pi = int(lo.argmax()); pr, pc = pi // g, pi % g
        r_up = (pr + 0.5) / g * gres; c = (pc + 0.5) / g * gres
        obj_px = np.array([R - 1 - r_up / (gres / R), c / (gres / R)], np.float32)
        return T, obj_px
    if args.binder == "molmo":
        # Molmo open-vocab POINTING VLM (point-to-the-X). Fully general, no proto/body_pos. Runs ONCE per episode.
        from bind_foveate import nm
        import re as _re, PIL.Image as _PI
        global _MOLMO
        if "_MOLMO" not in globals() or _MOLMO is None:
            from transformers import AutoModelForCausalLM, AutoProcessor
            mid = "allenai/Molmo-7B-D-0924"
            _pr = AutoProcessor.from_pretrained(mid, trust_remote_code=True, torch_dtype="auto")
            _md = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, torch_dtype=torch.float16).to(dev)  # no accelerate auto-map (Molmo remote code != tfm5)
            globals()["_MOLMO"] = (_pr, _md)
        mpr, mmd = globals()["_MOLMO"]
        R = args.res; HR = args.bind_res
        hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
        query = nm(T).replace("_", " ")
        with torch.no_grad():
            mi = mpr.process(images=[_PI.fromarray(hi)], text=f"Point to the {query}.")
            mi = {k: v.to(mmd.device).unsqueeze(0) for k, v in mi.items()}
            from transformers import GenerationConfig
            go = mmd.generate_from_batch(mi, GenerationConfig(max_new_tokens=80, stop_strings="<|endoftext|>"), tokenizer=mpr.tokenizer)
            txt = mpr.tokenizer.decode(go[0, mi["input_ids"].size(1):], skip_special_tokens=True)
        m = _re.search(r'x\d*="([\d.]+)"\s+y\d*="([\d.]+)"', txt)   # Molmo points = PERCENT of image dims
        if m:
            px_c = float(m.group(1)) / 100.0 * HR; px_r = float(m.group(2)) / 100.0 * HR   # HR upright (col,row)
            obj_px = np.array([R - 1 - px_r / (HR / R), px_c / (HR / R)], np.float32)
        return T, obj_px
    if args.binder == "gdino":
        # GroundingDINO open-vocab DETECTION (text -> box), the detector inside Grounded-SAM2. No TF, no body_pos.
        from bind_foveate import nm
        import PIL.Image as _PI
        global _GD
        if "_GD" not in globals() or _GD is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            mid = "IDEA-Research/grounding-dino-base"
            _gp = AutoProcessor.from_pretrained(mid)
            _gm = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()
            globals()["_GD"] = (_gp, _gm)
        gp, gm = globals()["_GD"]
        R = args.res; HR = args.bind_res
        hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
        query = nm(T).replace("_", " ").strip() + " ."   # GroundingDINO wants lowercase, period-terminated
        with torch.no_grad():
            inp = gp(images=_PI.fromarray(hi), text=query.lower(), return_tensors="pt").to(dev)
            out = gm(**inp)
            res = gp.post_process_grounded_object_detection(out, inp["input_ids"], threshold=0.15, text_threshold=0.15,
                                                            target_sizes=[(HR, HR)])[0]
        if len(res["scores"]):
            bi = int(torch.argmax(res["scores"])); bx = res["boxes"][bi].tolist()
            cyx = ((bx[1] + bx[3]) / 2.0, (bx[0] + bx[2]) / 2.0)
            obj_px = np.array([R - 1 - cyx[0] / (HR / R), cyx[1] / (HR / R)], np.float32)
        return T, obj_px
    if args.binder == "owl":
        # OWLv2 open-vocab DETECTION (text query -> box). Fully general: no proto bank, no body_pos, no SAM regions.
        from bind_foveate import nm
        global _OWL
        if "_OWL" not in globals() or _OWL is None:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            _p = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            _m = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
            globals()["_OWL"] = (_p, _m)
        proc, mdl = globals()["_OWL"]
        R = args.res; HR = args.bind_res; sc = HR / R
        import PIL.Image as _PI
        hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()   # upright
        query = nm(T).replace("_", " ")
        with torch.no_grad():
            inp = proc(text=[[f"a photo of a {query}", query]], images=_PI.fromarray(hi), return_tensors="pt").to(dev)
            out = mdl(**inp)
            tgt = torch.tensor([[HR, HR]], device=dev)
            res = proc.post_process_grounded_object_detection(out, threshold=0.02, target_sizes=tgt)[0]
        if len(res["scores"]):
            bi = int(torch.argmax(res["scores"]))
            bx = res["boxes"][bi].tolist()   # [x0,y0,x1,y1] in HR (upright) coords
            cyx = ((bx[1] + bx[3]) / 2.0, (bx[0] + bx[2]) / 2.0)   # (row,col) upright HR
            obj_px = np.array([R - 1 - cyx[0] / sc, cyx[1] / sc], np.float32)   # -> res projection frame
        return T, obj_px
    if args.binder == "dino":
        from bind_exemplar import proto_crop
        from dino_separability import dino_feat
        from bind_foveate import nm
        hi = sim.render(width=args.bind_res, height=args.bind_res, camera_name="agentview")
        up = np.asarray(hi)[::-1].copy(); proto = bank.get(nm(T)); feats = {}
        for o in graspables:
            if rb.get(o) is None: continue
            c = proto_crop(sim, up, args.bind_res, "agentview", rb[o], 60)
            if c is not None: feats[o] = dino_feat(c, dev)
        return (max(feats, key=lambda o: float(feats[o] @ proto)) if (feats and proto is not None) else T), None
    # sam: SAM proposes WHOLE-OBJECT regions at res (hi-res over-segments objects into fragments -> worse:
    # bind 18/30 vs 21/30) -> foveated-DINOv2 match on a HI-RES crop around each region -> region pixel. NO body_pos.
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
    if best is not None:
        obj_px = np.array([R - 1 - best[1], best[2]], np.float32)   # upright-row -> projection frame
    return T, obj_px


def _container_z(sim, cb, dm, args):
    """DE-HARDCODE the z_cont_center=0.035 basket constant: detect the container TOP z from DEPTH
    (80th-pct z of the container region) so it generalizes basket(rim~0.035)->plate(top~0.0)->any.
    Returns (z_top, cont_w_xy_z3): ray-plane xy at the detected z (parallax-free) + detected top z.
    Falls back to the class constant if depth fails. Gated by --cont-depth (motor must be retrained on it)."""
    cdet = depth_localize(sim, obj_pixel(sim, cb, args.res), dm, args.res, mode="container")
    zc = float(cdet[2]) if cdet is not None else args.z_cont_center
    return zc, ray_plane(sim, cb, zc, args.res)


def localize_targets(sim, obs, args, chosen, rb, cb, obj_px, image_tools=None):
    """3D localization: returns (obj_w, cont_w, rim_top). Honest depth/ray-plane or oracle body_pos."""
    if args.loc == "depthflip":
        import robosuite.utils.camera_utils as _cu
        dm = _cu.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]))[::-1].copy()
        obj_w = depth_localize(sim, (obj_px if obj_px is not None else obj_pixel(sim, rb[chosen], args.res)), dm, args.res, mode="object")
        if args.cont_depth:   # general depth container-top (de-hardcodes z_cont_center per-container)
            zc, cont_w = _container_z(sim, cb, dm, args); rim_top = zc
        else:
            cont_w = ray_plane(sim, cb, args.z_cont_center, args.res); rim_top = args.z_cont_center + args.rim_h
        if obj_w is None: obj_w = body_pos(sim, rb[chosen]).astype(np.float32)
        if args.obj_oracle_z: obj_w[2] = float(body_pos(sim, rb[chosen])[2])
        obj_w[2] += args.obj_z_off
    elif args.loc == "nodeloc":
        dm = node_flipped(sim, obs["agentview_depth"])
        obj_w = node_localize(sim, (obj_px if obj_px is not None else obj_pixel(sim, rb[chosen], args.res)), dm, args.res, win=12, z_mode="surface")
        if args.cont_depth:
            zc, cont_w = _container_z(sim, cb, dm, args); rim_top = zc
        else:
            cont_w = ray_plane(sim, cb, args.z_cont_center, args.res); rim_top = args.z_cont_center + args.rim_h
        if obj_w is None: obj_w = body_pos(sim, rb[chosen]).astype(np.float32)
        if args.obj_oracle_z: obj_w[2] = float(body_pos(sim, rb[chosen])[2])
        obj_w[2] += args.obj_z_off
    elif args.loc == "rayplane":
        obj_w = ray_plane(sim, rb[chosen], args.z_obj, args.res); cont_w = ray_plane(sim, cb, args.z_cont, args.res)
        rim_top = args.z_cont + args.rim_h
    else:
        obj_w = body_pos(sim, rb[chosen]).astype(np.float32); cont_w = body_pos(sim, cb).astype(np.float32)
        rim_top, _ = body_aabb_top(sim, sim.model.body_name2id(cb))
    return obj_w, cont_w, rim_top


def resolve_pair_target(sim, bf, obj_noun, relation):
    """Resolve a pair's grasp-object to a scene body via language noun + relational which-instance (no oracle)."""
    cands = scene_bodies(bf)
    To = resolve_noun(obj_noun, cands)
    if To is None:
        return None
    if relation and relation.startswith("instance-"):
        idx = int(relation.split("-")[1])
        same = [n for n in cands if len(set((obj_noun or "").split()) & _body_tokens(n)) >= 1]
        if idx < len(same): To = same[idx]
    elif relation:
        ch = select_instance(sim, obj_noun, relation, cands)
        if ch: To = ch
    return To


def run_seq_episode(env, sim, obs, args, pair_specs, bf, graspables,
                    net, fnet, vm, vs, cmf, csf, fsteps, dev, bank, sam_gen, image_tools):
    """N-SUBGOAL SEQUENCER (long-10): execute pick-place PAIRS in order with the SAME learned dual-goal motor.
    Advance to the next pair when the held object is released over its container. Returns (success, grasped_any).
    This is the dual-goal->N-goal payoff: one primitive, re-targeted + re-localized per pair (no fixed 2-subgoal cap)."""
    all_conts = list({ps["cont"] for ps in pair_specs})

    def setup(spec):
        To = resolve_pair_target(sim, bf, spec["obj"], spec.get("relation"))
        Co = resolve_noun(spec["cont"], scene_bodies(bf))
        rb = resolve_bodies(sim, graspables + all_conts); cb = rb.get(Co)
        if cb is None:
            cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and Co and Co in b]
            cb = cand[0] if cand else None
        if To is None or rb.get(To) is None or cb is None:
            return None
        chosen, obj_px = bind_target(sim, obs, args, To, graspables, rb, dev, bank, sam_gen)
        obj_w, cont_w, rim_top = localize_targets(sim, obs, args, chosen, rb, cb, obj_px)
        return {"T": To, "cb": cb, "rb": rb, "obj_w": obj_w, "cont_w": cont_w, "rim_top": rim_top,
                "cc": cont_class_from_name(spec.get("cont") or Co or "")}

    def reset_state(cur):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return dict(z0=float(body_pos(sim, cur["rb"][cur["T"]])[2]), lifted=0.0, close_run=0, open_run=0,
                    held=False, grasp_off=None, ci=0, chunk=None, released=0, ee_zmin=float(ee[2]))

    pidx = 0; cur = setup(pair_specs[0])
    if cur is None:
        return False, False, None
    st = reset_state(cur); grasped_any = False; REC = []
    for step in range(args.horizon * len(pair_specs)):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        obj_rel = cur["obj_w"] - ee; cont_rel = cur["cont_w"] - ee
        if (not st["held"]) or args.reflex:
            # ---- LEARNED motor (grasp + reflex=1 place) ----
            if st["chunk"] is None or st["ci"] >= args.replan:
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
                        chh = net(img, torch.tensor(vec)[None].to(dev), cc=torch.tensor([cur.get("cc", 0)], dtype=torch.long, device=dev)).cpu().numpy()[0]
                st["chunk"] = decanon_chunk(chh, theta) if phi else chh; st["ci"] = 0
            a = st["chunk"][st["ci"]]; st["ci"] += 1
            grip = 1.0 if a[6] > 0 else -1.0
            act = a[:7].copy(); act[6] = grip
        else:
            # ---- ANALYTIC grounded-physics place (offline DAgger teacher, reflex=0) ----
            tgt_z = cur["rim_top"] + args.clear
            if args.obj_aware_release: tgt_z = tgt_z - float(st["grasp_off"][2])
            des_ee = np.array([cur["cont_w"][0] - st["grasp_off"][0], cur["cont_w"][1] - st["grasp_off"][1], tgt_z], np.float32)
            over = float(np.linalg.norm((ee + st["grasp_off"])[:2] - cur["cont_w"][:2])) < args.tol and abs(ee[2] - tgt_z) < args.tol
            if over or st["released"]:
                st["released"] += 1; act = np.zeros(7, np.float32); act[6] = -1.0
            else:
                d = np.clip((des_ee - ee) * args.gain, -1.0, 1.0); act = np.zeros(7, np.float32); act[:3] = d; act[6] = 1.0
            grip = act[6]
        st["close_run"] = st["close_run"] + 1 if grip > 0 else 0
        st["open_run"] = st["open_run"] + 1 if grip < 0 else 0
        if args.logdir:   # DAgger: log (wrist, obj_rel, cont_rel, proprio, executed action) for distillation
            _wr = image_tools.resize_with_pad(np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]), args.img, args.img).astype(np.uint8)
            _pr = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])), np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
            REC.append((_wr, (cur["obj_w"] - ee).astype(np.float32), (cur["cont_w"] - ee).astype(np.float32), _pr, act[:7].astype(np.float32).copy()))
        obs, _, done, _ = env.step(act.tolist())
        ee_now = np.asarray(obs["robot0_eef_pos"], np.float32); st["ee_zmin"] = min(st["ee_zmin"], float(ee_now[2]))
        objp = body_pos(sim, cur["rb"][cur["T"]]).astype(np.float32)
        st["lifted"] = max(st["lifted"], float(objp[2] - st["z0"]))
        if (not st["held"]) and st["close_run"] > 8 and (float(ee_now[2]) - st["ee_zmin"] > args.grasp_lift):
            st["held"] = True; grasped_any = True; st["grasp_off"] = (cur["obj_w"] - ee_now)
        # pair completion: learned=released-over-container; analytic=release counter elapsed
        if args.reflex:
            placed = (st["held"] and st["open_run"] > 2 and
                      float(np.linalg.norm(objp[:2] - cur["cont_w"][:2])) < args.seq_rad and (objp[2] - cur["rim_top"]) < 0.08)
        else:
            placed = st["released"] > 8
        if placed and pidx + 1 < len(pair_specs):   # advance only if more pairs remain
            pidx += 1
            nxt = setup(pair_specs[pidx])
            if nxt is None:
                break
            cur = nxt; st = reset_state(cur); continue
        if (not args.reflex) and st["released"] > 25:   # last analytic place done -> let it settle a bit then stop
            break
        if done:
            break
    try: ok = bool(env.env._check_success())
    except Exception: ok = False
    return ok, grasped_any, (REC if (args.logdir and ok and len(REC) > 10) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_selector.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--flow", default="")   # path to FlowMotor ckpt -> use flow-matching head (learned-place fix) instead of L1
    p.add_argument("--lang-plan", type=int, default=0)   # derive grasp-obj + container from the LANGUAGE instruction (language_planner), NOT the BDDL goal / oracle
    p.add_argument("--seq", type=int, default=0)          # N-subgoal SEQUENCER (long-10): execute pick-place PAIRS in order, advance on completion (removes dual-goal)
    p.add_argument("--seq-rad", type=float, default=0.12) # xy radius (m) for detecting a pair placed over its container (advance trigger)
    p.add_argument("--logdir", default="")   # if set: log SUCCESSFUL episodes' (wrist,obj_rel,cont_rel,proprio,executed-chunk) -> analytic-DAgger place distillation
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--canon-mode", default="perstep_obj"); p.add_argument("--hide", type=int, default=1)
    p.add_argument("--clear", type=float, default=0.10)   # release height above rim
    p.add_argument("--gain", type=float, default=8.0); p.add_argument("--tol", type=float, default=0.025)
    p.add_argument("--reflex", type=int, default=0)       # 0=physics place, 1=pure reflex (baseline A/B)
    p.add_argument("--binder", default="oracle", choices=["oracle", "dino", "sam", "owl", "molmo", "gdino", "learned"])   # sam = SAM proposes pixels (no body_pos) -> FULLY honest attention
    p.add_argument("--sam-ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--ground-head", default="data/ground_head.pt")   # learned grounding-head ckpt for --binder learned
    p.add_argument("--loc", default="oracle", choices=["oracle", "rayplane", "depthflip", "nodeloc"])   # depthflip/nodeloc=honest depth localizer (nodeloc=general masked-depth, validated ~1.7cm)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.04)
    p.add_argument("--rim-h", type=float, default=0.075)   # honest container rim height above its table-plane z (basket ~7.5cm)
    p.add_argument("--z-cont-center", type=float, default=0.035)   # basket center height (class const) for ray-plane container loc (avoids depth parallax)
    p.add_argument("--cont-depth", type=int, default=0)            # DE-HARDCODE z_cont_center: detect container TOP z from DEPTH (general basket->plate->any); needs motor retrained on it
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
    _cond = "n_cls" in ck   # conditioned motor (container-class embedding) vs legacy head
    net = WristMotor(prop_dim=ck["prop_dim"], n_cls=ck.get("n_cls", len(CC_VOCAB)), conditioned=_cond).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
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
    if args.logdir: pathlib.Path(args.logdir).mkdir(parents=True, exist_ok=True)
    dag_n = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets and not args.seq: continue   # seq derives targets from the plan, not the BDDL
        stem = pathlib.Path(bf).stem; graspables = [o for o in objs if args.container not in o]
        import re as _re
        m = _re.match(r"pick_up_the_(.+)$", stem); T = targets[0] if targets else None
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for s in ("between", "next", "on", "from", "in", "and"):
                if s in toks: cut = min(cut, toks.index(s))
            tc = "_".join(toks[:cut]); cand = next((o for o in graspables if tc and tc in o), None)
            if cand: T = cand
        cont_name = args.container + "_1"; _relnoun = None; _relrel = None
        if args.lang_plan and not args.seq:   # DE-HARDCODE: grasp-obj + container from the LANGUAGE instruction (not BDDL goal / oracle)
            from language_planner import plan as _lplan
            _cands = scene_bodies(bf); _subs = _lplan(instr)
            _pp = next((s for s in _subs if s["skill"] == "place"), None)
            if _pp is None:
                print(f"  {stem[:30]:32s} skip (no pick-place subgoal; articulated)", flush=True); continue
            _T = resolve_noun(_pp["obj"], _cands); _C = resolve_noun(_pp["target"], _cands)
            if _T is None or _C is None:
                print(f"  {stem[:30]:32s} skip (noun unresolved {_pp['obj']}->{_T} / {_pp['target']}->{_C})", flush=True); continue
            T = _T; cont_name = _C; _relnoun = _pp["obj"]; _relrel = _pp.get("relation")
            if len(_subs) > 2: print(f"  {stem[:30]:32s} NOTE multi-subgoal ({len(_subs)}) -> first pair only (long-10 sequencer TODO)", flush=True)
        elif args.container == "auto":   # GOAL suite: route container per-task from the (:goal (On X Y)) predicate
            _gtxt = pathlib.Path(bf).read_text()
            _gm = _re.search(r"\(:goal.*?\((?:On|In)\s+(\w+)\s+(\w+)\)", _gtxt, _re.S)
            if _gm is None or _gm.group(2).startswith("main_table"):
                print(f"  {stem[:30]:32s} skip-articulated/push", flush=True); continue
            T = _gm.group(1); cont_name = _re.sub(r"(_[a-z]+)+_region$", "", _gm.group(2))
        pair_specs = None
        if args.seq:   # N-SUBGOAL SEQUENCER: group plan() subgoals into ordered pick-place PAIRS
            from language_planner import plan as _lplan
            _subs = _lplan(instr); pair_specs = []; _i = 0
            while _i < len(_subs):
                _s = _subs[_i]
                if _s["skill"] == "grasp" and _i + 1 < len(_subs) and _subs[_i + 1]["skill"] == "place":
                    pair_specs.append({"obj": _s["obj"], "cont": _subs[_i + 1]["target"], "relation": _s.get("relation")})
                    _i += 2
                else:
                    _i += 1   # articulated/unpaired subgoal -> not executed (task fails honestly)
            _nart = sum(1 for _s in _subs if _s["skill"] in ("open", "turnon", "close", "push"))
            print(f"  {stem[:30]:32s} SEQ {len(pair_specs)} pair(s)" + (f" +{_nart} articulated(unhandled)" if _nart else ""), flush=True)
            if not pair_specs:
                print(f"  {stem[:30]:32s} skip (no pick-place pair)", flush=True); continue
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
            if args.seq:   # N-subgoal sequencer episode (long-10) — same learned motor (or analytic teacher), re-targeted per pair
                _ok, _gr, _rec = run_seq_episode(env, sim, obs, args, pair_specs, bf, graspables,
                                                 net, fnet, vm, vs, cmf, csf, fsteps, dev, bank, sam_gen, image_tools)
                if _rec is not None:   # DAgger: save the SUCCESSFUL multi-pair trajectory as distillation npz
                    W = np.stack([r[0] for r in _rec]); O = np.stack([r[1] for r in _rec]); C = np.stack([r[2] for r in _rec])
                    P = np.stack([r[3] for r in _rec]); A = np.stack([r[4] for r in _rec]); Tn = len(A)
                    CH = np.zeros((Tn, 10, 7), np.float32)
                    for tt in range(Tn):
                        e = min(tt + 10, Tn); CH[tt, :e - tt] = A[tt:e]
                        if e - tt < 10: CH[tt, e - tt:] = A[Tn - 1]
                    np.savez_compressed(pathlib.Path(args.logdir) / f"{stem[:40]}_seq{dag_n}.npz",
                                        wrist=W.astype(np.uint8), obj_rel=O.astype(np.float32), cont_rel=C.astype(np.float32),
                                        proprio=P.astype(np.float32), chunk=CH); dag_n += 1
                succ.append(int(_ok)); grasp.append(int(_gr)); env.close(); continue
            if args.lang_plan and _relrel and not _relrel.startswith("instance-"):   # RELATIONAL which-instance (no oracle which-bowl)
                _ch = select_instance(sim, _relnoun, _relrel, scene_bodies(bf))
                if _ch: T = _ch
            rb = resolve_bodies(sim, graspables + [cont_name]); cb = rb.get(cont_name)
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and cont_name in b]
                cb = cand[0] if cand else None
            if rb.get(T) is None or cb is None: env.close(); succ.append(0); grasp.append(0); continue
            # ---- BIND target identity ON THE FULL SCENE (must be BEFORE hide!) ----
            chosen, obj_px = bind_target(sim, obs, args, T, graspables, rb, dev, bank, sam_gen)
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
            obj_w, cont_w, rim_top = localize_targets(sim, obs, args, chosen, rb, cb, obj_px)
            ee_z0 = float(np.asarray(obs["robot0_eef_pos"], np.float32)[2]); ee_zmin = ee_z0
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; close_run = 0; held = False; grasp_off = None; released = 0
            chunk = None; ci = 0; REC = []
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
                                chh = net(img, torch.tensor(vec)[None].to(dev), cc=torch.tensor([cont_class_from_name(cont_name)], dtype=torch.long, device=dev)).cpu().numpy()[0]
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
                if args.logdir:
                    _wr = image_tools.resize_with_pad(np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    _pr = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])), np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    REC.append((_wr, (obj_w - ee).astype(np.float32), (cont_w - ee).astype(np.float32), _pr, act[:7].astype(np.float32).copy()))
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
            if args.logdir and ok and len(REC) > 10:
                W = np.stack([r[0] for r in REC]); O = np.stack([r[1] for r in REC]); C = np.stack([r[2] for r in REC])
                P = np.stack([r[3] for r in REC]); A = np.stack([r[4] for r in REC]); Tn = len(A)
                CH = np.zeros((Tn, 10, 7), np.float32)
                for t in range(Tn):
                    e = min(t + 10, Tn); CH[t, :e - t] = A[t:e]
                    if e - t < 10: CH[t, e - t:] = A[Tn - 1]
                np.savez_compressed(pathlib.Path(args.logdir) / f"{stem[:40]}_dag{dag_n}.npz",
                                    wrist=W.astype(np.uint8), obj_rel=O.astype(np.float32), cont_rel=C.astype(np.float32),
                                    proprio=P.astype(np.float32), chunk=CH); dag_n += 1
            succ.append(int(ok)); grasp.append(int(lifted > 0.03)); placed_phys.append(int(held and not args.reflex)); env.close()
        print(f"  {stem[:30]:32s} succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp={np.mean(grasp):.2f}", flush=True)
    print(f"\n=== PHYSICS-PLACE [reflex={args.reflex}] on {pathlib.Path(args.bddl_dir).name}: "
          f"succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp_rate={np.mean(grasp):.3f}  bind={bind_ok}/{nb} ===  [reflex-place baseline ~0.37]", flush=True)
    print("PHYSPLACE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
