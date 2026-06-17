"""LIBERO-10 (long-horizon) via the SEQUENCER: split the instruction into subgoals; route each by verb.
'put_both_the_A_and_the_B_in_the_basket' -> TWO sequential specialist pick-places with RE-PERCEPTION between subgoals
(honest verification: subgoal k done when object k localized at the basket). Other verbs -> frozen pi0.5 with the full
instruction (trained on these tasks). Grocery catalog from the object-suite lan dir (same classes).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/eval_10.py \
       --bddl-dir <libero_10_object> --init-dir <...> --proto-dir <object_lan> --proto-init-dir <object_lan_init> \
       --n 10 --trials 1 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle, build_obs
from train_wristcam_motor import WristMotor
from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
from bind_exemplar import build_bank
from bind_foveate import nm
from eval_e2e_depth import depth_localize


def both_targets_from_stem(stem):
    """'..._put_both_the_A_and_the_B_in_the_basket' -> (A, B) class strings, else None."""
    m = re.search(r"put_both_the_(.+?)_and_the_(.+?)_in_the_basket", stem)
    return (m.group(1), m.group(2)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head_canon_bowls.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--proto-dir", required=True); p.add_argument("--proto-init-dir", required=True)
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--bind-res", type=int, default=1024)
    p.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    p.add_argument("--config-name", default="pi05_libero"); p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=1)
    p.add_argument("--sub-horizon", type=int, default=340); p.add_argument("--pi05-horizon", type=int, default=600); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--z-obj", type=float, default=0.015); p.add_argument("--z-cont", type=float, default=0.05)
    p.add_argument("--sam-points", type=int, default=32); p.add_argument("--sam-minarea", type=int, default=25)
    p.add_argument("--lift-secure", type=float, default=0.06); p.add_argument("--hold-grip", type=int, default=1)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda"; R = args.res; HR = args.bind_res
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval(); vm, vs = ck["vm"], ck["vs"]
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.sam_points, min_mask_region_area=args.sam_minarea)
    print("building grocery catalog (object-suite refs)...", flush=True)
    bank = build_bank(sorted(glob.glob(str(pathlib.Path(args.proto_dir) / "*.bddl"))),
                      args.proto_init_dir, [int(x) for x in args.proto_inits.split(",")], HR, "agentview", dev)
    print("loading pi0.5 (router fallback for non-pick-place verbs)...", flush=True)
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)

    succ = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        both = both_targets_from_stem(stem)
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            if both is None:
                # ---- non-pick-place long-horizon: frozen pi0.5 with the full instruction ----
                chunk = None; ci = 0
                for step in range(args.pi05_horizon):
                    if chunk is None or ci >= args.replan:
                        o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                         obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                        chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                    obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                    if done: break
                try: ok = bool(env.env._check_success())
                except Exception: ok = False
                succ.append(int(ok)); env.close()
                print(f"    [{stem[:34]:36s} t{t}] route=pi05 ok={int(ok)}", flush=True)
                continue
            # ---- SEQUENCER: two specialist pick-places with re-perception between subgoals ----
            subok = []
            for k, cls_k in enumerate(both):
                # match instance name for this subgoal (for localization identity via catalog key)
                Tk = next((o for o in objs if cls_k in o), None)
                if Tk is None: break
                state = np.asarray(sim.get_state().flatten())
                env.close()
                r = depth_localize(bf, state, Tk, bank, gen, dev, R, HR, args.z_obj, args.z_cont, args.container)
                obj_w, cont_w = r[0], r[1]
                env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
                env.seed(args.seed + ti); env.reset(); sim = env.env.sim
                obs = env.set_init_state(state)
                if obj_w is None: break
                if float(np.linalg.norm(obj_w[:2] - cont_w[:2])) < 0.12:
                    subok.append(True); continue   # already there (honest check)
                rb = resolve_bodies(sim, [Tk]); z0 = body_pos(sim, rb[Tk])[2]
                lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
                for step in range(args.sub_horizon):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32)
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
                        with torch.no_grad(): chh = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                        chunk = decanon_chunk(chh, theta); ci = 0
                    a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                    close_cnt = close_cnt + 1 if grip > 0 else 0
                    if (not held) and close_cnt > 8 and lifted > args.lift_secure: held = True
                    if args.hold_grip and held and grip < 0 and float(np.linalg.norm(ee[:2] - cont_w[:2])) > 0.12: grip = 1.0
                    act = a[:7].copy(); act[6] = grip
                    obs, _, done, _ = env.step(act.tolist())
                    lifted = max(lifted, body_pos(sim, rb[Tk])[2] - z0)
                    if done: break
                # honest subgoal check
                d_end = float(np.linalg.norm(body_pos(sim, rb[Tk])[:2] - cont_w[:2]))
                subok.append(d_end < 0.12)
                # go-home before next subgoal
                HOME = np.array([-0.15, 0.0, 0.30], np.float32)
                for _ in range(25):
                    ee = np.asarray(obs["robot0_eef_pos"], np.float32); d = HOME - ee
                    if float(np.linalg.norm(d)) < 0.04: break
                    aa = np.clip(d * 6.0, -0.9, 0.9)
                    obs, _, _, _ = env.step([float(aa[0]), float(aa[1]), float(aa[2]), 0.0, 0.0, 0.0, -1.0])
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); env.close()
            print(f"    [{stem[:34]:36s} t{t}] route=SEQ sub={subok} ok={int(ok)}", flush=True)
        print(f"  {stem[:40]:42s} so_far={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})", flush=True)
    print(f"\n=== LIBERO-10 SEQUENCER on {pathlib.Path(args.bddl_dir).name}: {np.mean(succ):.3f} ({sum(succ)}/{len(succ)}) ===  [pi0.5 cells 21.5/11, VLS 25.5/15.5]", flush=True)
    print("E2E10_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
