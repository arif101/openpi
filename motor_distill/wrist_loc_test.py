"""FOUNDATIONAL test for the closed-loop wrist servo: when the coarse goal brings the gripper CLOSE, can the WRIST
eye-in-hand camera localize the object PRECISELY (<<1cm) -- vs the ~3cm overhead agentview? Drives the existing motor
toward the (privileged) object for a reach, then at the close state renders the wrist cam, finds the object (SAM+DINOv2
or, as an upper bound, the projected center), and ray-planes in the WRIST frame to get 3D. Reports wrist-localization
error vs true. If <<1cm, reach-then-wrist-relocalize works (reuse the 0.50 motor with a wrist-refined goal).
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="data/motor_head_canon_z.pt")
    ap.add_argument("--bddl-dir", default="/root/LIBERO-Pro-data/bddl_files/libero_object_swap")
    ap.add_argument("--init-dir", default="/root/LIBERO-Pro-data/init_files/libero_object_swap")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--reach-steps", type=int, default=120); ap.add_argument("--wcam", default="robot0_eye_in_hand")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from canon import canon_angle, rot_vec_xy, rot_image, decanon_chunk
    from train_wristcam_motor import WristMotor
    dev = "cuda"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotor(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval(); vm, vs = ck["vm"], ck["vs"]
    R = args.res
    agv_err, wr_err, wr_px, in_fov = [], [], [], []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T]); obj = body_pos(sim, rb[T])
        # reach: drive the motor toward the privileged object goal (just to get CLOSE)
        chunk = None; ci = 0
        from openpi_client import image_tools
        for step in range(args.reach_steps):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32); goal_rel = (body_pos(sim, rb[T]).astype(np.float32) - ee)
            if chunk is None or ci >= 5:
                wr_raw = np.asarray(obs["robot0_eye_in_hand_image"]); wim = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), 128, 128).astype(np.uint8)
                prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])), np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                theta = canon_angle(goal_rel); phi = -theta
                wim2 = rot_image(wim, phi); g_in = rot_vec_xy(goal_rel, phi); pr = prop.copy(); pr[:3] = rot_vec_xy(pr[:3], phi)
                img = torch.tensor(np.transpose(wim2.astype(np.float32) / 255, (2, 0, 1)))[None].to(dev)
                vec = ((np.concatenate([g_in, pr, [0.0]]).astype(np.float32) - vm) / vs).astype(np.float32)
                with torch.no_grad(): ch = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                chunk = decanon_chunk(ch, theta); ci = 0
            a = chunk[ci].copy(); ci += 1; a[6] = -1.0; obs, _, done, _ = env.step(a[:7].tolist())
            ee = np.asarray(obs["robot0_eef_pos"]);
            if np.linalg.norm(ee[:2] - body_pos(sim, rb[T])[:2]) < 0.04: break   # close enough -> stop reach
        # --- at the close state: localize the object in the WRIST camera ---
        obj = body_pos(sim, rb[T]); ee = np.asarray(obs["robot0_eef_pos"])
        dxy = float(np.linalg.norm(ee[:2] - obj[:2]))
        # project the object into the wrist cam; is it in FOV? how big? localize via wrist ray-plane at table z
        w2p_w = cu.get_camera_transform_matrix(sim, args.wcam, R, R)
        px = cu.project_points_from_world_to_camera(obj[None], w2p_w, R, R)[0]
        infov = (0 <= px[0] < R) and (0 <= px[1] < R); in_fov.append(int(infov))
        # apparent size: project a point 2cm across at the object
        px2 = cu.project_points_from_world_to_camera((obj + np.array([0.02, 0, 0]))[None], w2p_w, R, R)[0]
        size_px = float(np.linalg.norm(px2 - px)); wr_px.append(size_px)
        # wrist localization: ray through the object pixel ∩ table plane z=0
        inv = np.linalg.inv(w2p_w); d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
        r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
        q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0])
        q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0])
        ray = q2 - q1; tt = (0.0 - q1[2]) / ray[2]; pw = q1 + tt * ray
        wr_err.append(float(np.linalg.norm(pw[:2] - obj[:2])))
        # agentview localization (the ~3cm baseline) for comparison
        w2p_a = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        pxa = cu.project_points_from_world_to_camera(obj[None], w2p_a, R, R)[0]
        agv_err.append(0.2 / 100)  # body_pos projection is exact; the 3cm was the SAM-mask offset (not measured here)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} reach_dxy={dxy*100:4.1f}cm  obj_in_wrist_fov={infov}  obj_size_wrist={size_px:4.0f}px(2cm)  wrist_loc_err={wr_err[-1]*100:.2f}cm", flush=True)
        env.close()
    print(f"\n=== WRIST localization when close: in-FOV {int(np.mean(in_fov)*100)}%, obj apparent ~{np.mean(wr_px):.0f}px/2cm (agentview ~20px/whole), wrist_loc_err mean={np.mean(wr_err)*100:.2f}cm ===", flush=True)
    print("WRISTLOC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
