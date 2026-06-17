"""GENERALIST-FIRST, SPECIALIST-RECOVERY cascade: run frozen pi0.5 (memorized-strong on standard layouts) for phase A;
verify with OUR honest perception (depth-localize target vs basket at the CURRENT state); if not done, switch mid-episode
to the factored pipeline (re-localize NOW -> canon motor) for recovery. No oracle predicates in the router; no
memorization assumptions. Complementary strengths: pi0.5 lan~1.0 + ours swap 0.53 -> every cell >= max(parts).
State round-trip avoids the EGL depth-corruption: capture sim state -> close env -> depth env localizes at state ->
fresh env resumes at state.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv/bin/python motor_distill/eval_cascade.py \
       --bddl-dir <axis> --init-dir <axis> --proto-dir <axis> --proto-init-dir <axis> --n 10 --trials 1 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle, build_obs
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from bind_exemplar import build_bank
from eval_e2e_depth import depth_localize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_z.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", required=True); p.add_argument("--proto-init-dir", required=True)
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=1)
    p.add_argument("--phase-a", type=int, default=200); p.add_argument("--phase-b", type=int, default=300); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.05); p.add_argument("--z-ref-init", type=int, default=35)
    p.add_argument("--sam-points", type=int, default=32); p.add_argument("--sam-minarea", type=int, default=25)
    p.add_argument("--done-thresh", type=float, default=0.12)   # target within this xy of basket => declare done (honest check)
    p.add_argument("--lift-secure", type=float, default=0.06); p.add_argument("--hold-grip", type=int, default=1)
    p.add_argument("--order", default="generalist-first", choices=["generalist-first", "specialist-first", "auto"])
    p.add_argument("--reloc-thresh", type=float, default=0.08)   # auto-router: target >this from canonical (catalog-ref) xy => relocated => specialist-first
    p.add_argument("--canon-bddl-dir", default=""); p.add_argument("--canon-init-dir", default="")   # STANDARD-layout dir for canonical positions (e.g. the lan axis)
    p.add_argument("--phase-a-pi05", type=int, default=200)      # short leash for the generalist when it goes first (it succeeds fast or wrecks)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda"; R = args.res; HR = args.bind_res
    # our components
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval(); vm, vs = ck["vm"], ck["vs"]
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.sam_points, min_mask_region_area=args.sam_minarea)
    print("building DINOv2 bank...", flush=True)
    pbddls = sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl")))
    bank = build_bank(pbddls, args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], HR, "agentview", dev)
    # pi0.5 (frozen generalist)
    print("loading pi0.5...", flush=True)
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)

    succ = []; used_recovery = 0; rec_succ = 0; nb = 0
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
        def run_pi05(env, obs, steps):
            chunk = None; ci = 0; done = False
            for _ in range(steps):
                if chunk is None or ci >= args.replan:
                    o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                if done: break
            return obs, done

        def run_specialist(env, obs, sim, rbT_body, obj_w, cont_w, steps):
            z0 = body_pos(sim, rbT_body)[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0; done = False
            for _ in range(steps):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
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
                if (not held) and close_cnt > 8 and lifted > args.lift_secure: held = True
                if args.hold_grip and held and grip < 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.12: grip = 1.0
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rbT_body)[2] - z0)
                if done: break
            return obs, done

        # auto-router: canonical target xy from the STANDARD-layout dir's ref scene (same stem), via OUR honest localizer
        canon_xy = None
        if args.order == "auto":
            try:
                cb_dir = args.canon_bddl_dir or args.bddl_dir; ci_dir = args.canon_init_dir or args.init_dir
                cbf = pathlib.Path(cb_dir) / f"{stem}.bddl"; cfi = pathlib.Path(ci_dir) / f"{stem}.pruned_init"
                if cbf.exists() and cfi.exists():
                    cinits = np.asarray(torch.load(cfi, weights_only=False))
                    rc = depth_localize(str(cbf), cinits[int(args.proto_inits.split(",")[0])], T, bank, gen, dev, R, HR, z_obj_task, args.z_cont, args.container)
                    if rc[0] is not None: canon_xy = rc[0][:2]
            except Exception as e:
                print(f"  canonical lookup failed ({type(e).__name__}) -> router defaults to PI05-first", flush=True)
        for t in range(nt):
            ti = s0 + t; nb += 1
            if args.order == "auto":
                try:
                    r = depth_localize(bf, inits[ti], T, bank, gen, dev, R, HR, z_obj_task, args.z_cont, args.container)
                except Exception:
                    r = (None, None, "__err__")
                relocated = (r[0] is not None and canon_xy is not None and float(np.linalg.norm(r[0][:2] - canon_xy)) > args.reloc_thresh)
                spec_first = bool(relocated)
                print(f"    [{stem[:18]:18s} t{t}] router: relocated={int(relocated)} -> {'SPEC' if spec_first else 'PI05'}-first", flush=True)
            else:
                spec_first = args.order == "specialist-first"
            # ---------- PHASE A ----------
            if spec_first and args.order != "auto":
                r = depth_localize(bf, inits[ti], T, bank, gen, dev, R, HR, z_obj_task, args.z_cont, args.container)
            if spec_first:
                obj_w, cont_w, chosen = r[0], r[1], r[2]
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, list(dict.fromkeys(graspables + [T])))
            if spec_first:
                if r[0] is None: obs, done = obs, False
                else: obs, done = run_specialist(env, obs, sim, rb[T], obj_w, cont_w, args.phase_a)
            else:
                obs, done = run_pi05(env, obs, args.phase_a_pi05)
            try: ok_a = bool(env.env._check_success())
            except Exception: ok_a = False
            if done or ok_a:
                env.close()
                succ.append(int(ok_a)); print(f"    [{stem[:18]:18s} t{t}] phaseA({args.order[:4]}) done -> ok={int(ok_a)}", flush=True); continue
            # RETRACT FIRST (clear the arm out of the camera's view of the objects), THEN capture state + localize.
            # (Arm-occlusion at the captured state was breaking every specialist recovery: localization saw the arm.)
            for _ in range(16):
                obs, _, _, _ = env.step([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, -1.0])
            state = np.asarray(sim.get_state().flatten())
            env.close()
            # ---------- HONEST CHECK at the retracted state ----------
            r2 = depth_localize(bf, state, T, bank, gen, dev, R, HR, z_obj_task, args.z_cont, args.container)
            obj_w2, cont_w2 = r2[0], r2[1]
            if obj_w2 is not None and float(np.linalg.norm(obj_w2[:2] - cont_w2[:2])) < args.done_thresh:
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
                env.seed(args.seed + ti); env.reset(); env.set_init_state(state)
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                env.close(); succ.append(int(ok))
                print(f"    [{stem[:18]:18s} t{t}] check: at-basket -> ok={int(ok)}", flush=True); continue
            used_recovery += 1
            # ---------- PHASE B (the other policy) from the captured state ----------
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(state)
            rb = resolve_bodies(sim, list(dict.fromkeys(graspables + [T])))
            HOME = np.array([-0.15, 0.0, 0.30], np.float32)   # canon motor's training start pose (approx)
            for _ in range(30):   # go-home reflex: spec recovery must start from an in-distribution pose
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                d = HOME - ee
                if float(np.linalg.norm(d)) < 0.04: break
                a = np.clip(d * 6.0, -0.9, 0.9)
                obs, _, _, _ = env.step([float(a[0]), float(a[1]), float(a[2]), 0.0, 0.0, 0.0, -1.0])
            if spec_first:
                obs, done = run_pi05(env, obs, args.phase_b)
            else:
                if obj_w2 is None: env.close(); succ.append(0); print(f"    [{stem[:18]:18s} t{t}] recovery: no loc -> 0", flush=True); continue
                obs, done = run_specialist(env, obs, sim, rb[T], obj_w2, cont_w2, args.phase_b)
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); rec_succ += int(ok); env.close()
            print(f"    [{stem[:18]:18s} t{t}] RECOVERY({'pi05' if spec_first else 'spec'}) -> ok={int(ok)}", flush=True)
        print(f"  {stem[:36]:38s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== CASCADE (pi0.5 -> honest check -> factored recovery) on {pathlib.Path(args.bddl_dir).name}: "
          f"{np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  recovery_used={used_recovery}/{nb} recovery_succ={rec_succ} ===", flush=True)
    print("CASCADE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
