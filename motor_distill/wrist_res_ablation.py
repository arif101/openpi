"""GO/NO-GO for the committed plan: does the WRIST (eye-in-hand) camera give PRECISE 3D localization as the gripper
APPROACHES, vs the far AGENTVIEW one-shot (~3-6cm)? This validates/kills the 'wrist-cam resolution handoff' thesis
that is the source of <2cm precision -- BEFORE any tracker/LoRA spend.

Method (idealized tracker = instance-seg; measures LOCALIZATION precision, not binding): roll the motor toward the
privileged goal; each replan, localize the target object's 3D from BOTH cameras (project true pos -> seg id at that
pixel -> object mask centroid + median depth -> unproject) and log 3D error vs the camera-to-object distance.
DECISIVE: plot/aggregate agentview-err vs wrist-err binned by EE->object distance. If wrist-err drops <2cm at close
range while agentview stays ~3-6cm, the handoff thesis holds.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/wrist_res_ablation.py \
       --head data/motor_head.pt --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object \
       --init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object --n 10 --trials 2 --res 512
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle
from train_wristcam_motor import WristMotor


def loc_from_cam(sim, cam, R, true_pos, depth_norm, seg):
    """3D of the object at true_pos as seen by `cam`: project -> seg id there -> mask centroid + median depth -> 3D.
    Returns (err_cm, None) if object not visible in this camera."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); cam2world = np.linalg.inv(w2p)
    px = cu.project_points_from_world_to_camera(true_pos[None], w2p, R, R)[0]
    r = int(round(px[0])); c = int(round(px[1]))
    if not (0 <= r < R and 0 <= c < R): return None
    sid = seg.reshape(R, R)[r, c]
    if sid == 0: return None                                   # object not visible / background here
    m = (seg.reshape(R, R) == sid)
    if m.sum() < 10: return None
    ys, xs = np.where(m); rr, cc = ys.mean(), xs.mean()
    real = cu.get_real_depth_map(sim, depth_norm)
    pts = cu.transform_from_pixels_to_world(np.array([[rr, cc]]), real[None], cam2world)
    return float(np.linalg.norm(pts[0] - true_pos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=2)
    p.add_argument("--horizon", type=int, default=200); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--seed", type=int, default=20)
    p.add_argument("--res", type=int, default=512); p.add_argument("--img", type=int, default=128)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    AGV, WRI = "agentview", "robot0_eye_in_hand"
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    # accumulate (distance_cm, agv_err_cm, wri_err_cm-or-nan)
    rows = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False)) if args.init_dir else None
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True,
                                     camera_segmentations="instance")
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, [T]);
            if rb[T] is None: env.close(); continue
            chunk = None; ci = 0; held = False; close_cnt = 0; lifted = 0.0; z0 = body_pos(sim, rb[T])[2]
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32); true = body_pos(sim, rb[T])
                dist = float(np.linalg.norm(true - ee))
                if step % args.replan == 0 and not held:           # measure localization pre-grasp (object static)
                    a_err = loc_from_cam(sim, AGV, R, true, np.asarray(obs[AGV + "_depth"]), np.asarray(obs[AGV + "_segmentation_instance"]))
                    w_err = loc_from_cam(sim, WRI, R, true, np.asarray(obs[WRI + "_depth"]), np.asarray(obs[WRI + "_segmentation_instance"]))
                    rows.append((dist, a_err, w_err))
                if chunk is None or ci >= args.replan:
                    goal_rel = (true.astype(np.float32) - ee)
                    wr_raw = np.asarray(obs[WRI + "_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img)
                    img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    vec = ((np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    with torch.no_grad():
                        chunk = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]; ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                obs, _, done, _ = env.step(a[:7].tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if done or held: break
            env.close()
        print(f"  {stem[:28]:30s} samples={len(rows)}", flush=True)
    rows = np.array([(d, a if a is not None else np.nan, w if w is not None else np.nan) for d, a, w in rows])
    bins = [(0, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 1.0)]
    print("\n=== 3D LOCALIZATION ERROR vs EE->object distance (agentview far-view vs WRIST eye-in-hand) ===", flush=True)
    print(f"{'dist range':>14} | {'n':>4} | {'agv_err_cm':>10} | {'wrist_err_cm':>12} | {'wrist_vis%':>10}", flush=True)
    for lo, hi in bins:
        sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
        if len(sel) == 0: continue
        a_m = np.nanmean(sel[:, 1]) * 100; w_m = np.nanmean(sel[:, 2]) * 100
        wvis = np.mean(~np.isnan(sel[:, 2])) * 100
        print(f"{lo*100:5.0f}-{hi*100:3.0f}cm   | {len(sel):4d} | {a_m:10.1f} | {w_m:12.1f} | {wvis:9.0f}%", flush=True)
    print("WRIST_RES_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
