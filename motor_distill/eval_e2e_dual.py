"""DUAL-GOAL, NO-STATE-MACHINE eval of the factored wrist motor. The motor sees BOTH goals (obj_rel, cont_rel) every
step + the wrist image + proprio, and predicts the FULL action chunk including the gripper. There is NO held flag, NO
goal-switching, NO lift-secure, NO hold-grip heuristic -- the policy learned grasp->transport->release + gripper timing
from the teacher's actions, reading the phase from the wrist image. Per-episode SE(2) canon = object-initial bearing.

Localization:
  --loc oracle    : depth surface_point of the (binder-chosen) object + container  (SKILL-ISOLATION gate: motor alone)
  --loc rayplane  : depth-free pixel-ray x table-plane intersection                (HONEST deploy)
Binder: --binder oracle (use GT identity) | dino (foveated DINOv2 prototype match).

Run (oracle-isolation gate): PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python \
   motor_distill/eval_e2e_dual.py --head data/motor_head_dual_canon.pt --bddl-dir <d> --init-dir <d> \
   --binder oracle --loc oracle --container basket --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from collect_motor_data import surface_point


def ray_plane(sim, body, z_plane, R, cam="agentview"):
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]
    r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0])
    ray = q2 - q1; t = (z_plane - q1[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0
    pt = q1 + t * ray; return np.array([pt[0], pt[1], z_plane], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_dual_canon.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--binder", default="oracle", choices=["oracle", "dino"])
    p.add_argument("--loc", default="oracle", choices=["oracle", "body", "rayplane"])   # body = exact center (true oracle)
    p.add_argument("--canon-mode", default="perstep_obj", choices=["episode", "perstep_obj", "gripkey", "none"])
    p.add_argument("--grip-latch", type=int, default=0)   # DIAGNOSTIC: once closed K steps, force-hold grip (tests gripper-drop hypothesis)
    p.add_argument("--hide", type=int, default=1)          # hide distractors after binding (binder->HIDE->motor pipeline; matches hide-collected training => train==test)
    p.add_argument("--phase-latch", type=int, default=0)   # LATCH the learned phase: once alpha>0.5 for K consec replans, stay in place phase (kills flicker; grasp is monotone)
    p.add_argument("--priv-switch", type=int, default=0)   # DIAGNOSTIC: drive phase by PRIVILEGED held (grip>8 & object lift>2cm) instead of learned alpha -> isolates switch-accuracy vs place-precision
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.05)
    p.add_argument("--proto-dir", default=""); p.add_argument("--proto-init-dir", default="")
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    bank = None
    if args.binder == "dino":
        from bind_exemplar import build_bank
        print("building DINOv2 prototype bank...", flush=True)
        bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                          args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], args.bind_res, "agentview", dev)
    depth = (args.loc == "oracle")
    GRIP_OPEN_SEP = 0.05   # finger-separation sensor: > this = gripper OPEN (matches make_canon)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"E2E-DUAL (no state machine) binder={args.binder} loc={args.loc} canon={args.canon_mode}", flush=True)
    succ = []; grasp = []; switched = []; nflip = []; bind_ok = 0; nb = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem
        graspables = [o for o in objs if args.container not in o]
        import re as _re
        m = _re.match(r"pick_up_the_(.+)$", stem); T = targets[0]
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for stop in ("between", "next", "on", "from", "in", "and"):
                if stop in toks: cut = min(cut, toks.index(stop))
            tc = "_".join(toks[:cut])
            cand = next((o for o in graspables if tc and tc in o), None)
            if cand is not None: T = cand
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=depth)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables + [args.container + "_1"]); cb = rb[args.container + "_1"]
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            if rb.get(T) is None or cb is None: env.close(); succ.append(0); continue
            nb += 1
            if args.binder == "oracle":
                chosen = T
            else:
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
            bind_ok += int(chosen == T)
            if args.hide:   # binder->HIDE->motor: render-hide every non-target graspable (matches hide-collected training)
                for o in graspables:
                    if o == chosen or rb.get(o) is None: continue
                    bid = sim.model.body_name2id(rb[o])
                    for g in range(sim.model.ngeom):
                        if sim.model.geom_bodyid[g] == bid: sim.model.geom_rgba[g, 3] = 0.0
            # localize BOTH goals once at episode start
            if args.loc == "oracle":
                dmap = np.asarray(obs["agentview_depth"])
                obj_w = surface_point(sim, rb[chosen], dmap, args.res); cont_w = surface_point(sim, cb, dmap, args.res)
            elif args.loc == "body":
                obj_w = body_pos(sim, rb[chosen]).astype(np.float32); cont_w = body_pos(sim, cb).astype(np.float32)
            else:
                obj_w = ray_plane(sim, rb[chosen], args.z_obj, args.res); cont_w = ray_plane(sim, cb, args.z_cont, args.res)
            ee0 = np.asarray(obs["robot0_eef_pos"], np.float32)
            ep_phi = -canon_angle(obj_w - ee0) if args.canon_mode == "episode" else None   # PER-EPISODE fixed
            z0 = body_pos(sim, rb[chosen])[2]; lifted = 0.0; latched = False; close_run = 0   # grasp/place instrumentation
            a_seq = []   # alpha (phase) trajectory for this rollout
            placed = False; place_run = 0; held_priv = False   # phase latch / privileged-switch state
            chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    obj_rel = (obj_w.astype(np.float32) - ee); cont_rel = (cont_w.astype(np.float32) - ee)
                    # SE(2) canon frame for THIS step (parameter-free; see make_canon modes)
                    if args.canon_mode == "none":
                        phi = 0.0
                    elif args.canon_mode == "episode":
                        phi = ep_phi
                    elif args.canon_mode == "perstep_obj":
                        phi = -canon_angle(obj_rel)
                    else:   # gripkey: object bearing while OPEN, container while CLOSED (gripper sensor)
                        gopen = float(prop[3] - prop[4]) > GRIP_OPEN_SEP
                        phi = -canon_angle(obj_rel if gopen else cont_rel)
                    theta = -phi
                    if phi:
                        wr_in = rot_image(wr, phi)
                        o_in = rot_vec_xy(obj_rel, phi); c_in = rot_vec_xy(cont_rel, phi)
                        pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                    else:
                        wr_in = wr; o_in = obj_rel; c_in = cont_rel; pr = prop
                    img = torch.tensor(np.transpose(wr_in.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    vec = ((np.concatenate([o_in, c_in, pr]).astype(np.float32) - vm) / vs).astype(np.float32)
                    vec_t = torch.tensor(vec)[None].to(dev)
                    with torch.no_grad():
                        out_c, out_a = net(img, vec_t, return_phase=True)
                        alpha = float(torch.sigmoid(out_a)[0]); a_seq.append(alpha)
                        if args.priv_switch:
                            sel_ov = torch.tensor([1.0 if held_priv else 0.0], device=dev)
                            chh = net(img, vec_t, phase_override=sel_ov).cpu().numpy()[0]
                        elif args.phase_latch:
                            place_run = place_run + 1 if alpha > 0.5 else 0
                            if place_run >= args.phase_latch: placed = True
                            sel_ov = torch.tensor([1.0 if (placed or alpha > 0.5) else 0.0], device=dev)
                            chh = net(img, vec_t, phase_override=sel_ov).cpu().numpy()[0]
                        else:
                            chh = out_c.cpu().numpy()[0]
                    chunk = decanon_chunk(chh, theta) if phi else chh; ci = 0
                a = chunk[ci]; ci += 1
                g = 1.0 if a[6] > 0 else -1.0
                close_run = close_run + 1 if g > 0 else 0
                if args.grip_latch and close_run >= 3: latched = True
                act = a[:7].copy(); act[6] = 1.0 if latched else g
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, float(body_pos(sim, rb[chosen])[2] - z0))
                if close_run > 8 and lifted > 0.02: held_priv = True   # privileged switch signal (grip + object lift)
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); grasp.append(int(lifted > 0.03))
            ab = (np.asarray(a_seq) > 0.5).astype(int)
            switched.append(int(ab.any())); nflip.append(int(np.abs(np.diff(ab)).sum()) if len(ab) > 1 else 0)
            env.close()
        print(f"  {stem[:30]:32s} succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp={np.mean(grasp):.2f}  "
              f"phase_switched={np.mean(switched):.2f} flips={np.mean(nflip):.1f}  bind={bind_ok}/{nb}", flush=True)
    print(f"\n=== E2E-DUAL [{args.binder}/{args.loc}] latch={args.grip_latch} on {pathlib.Path(args.bddl_dir).name}: "
          f"succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp_rate={np.mean(grasp):.3f}  bind_ok={bind_ok}/{nb} ===  [swap: pi0.5 17% | VLS 36.81%]", flush=True)
    print("E2EDUAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
