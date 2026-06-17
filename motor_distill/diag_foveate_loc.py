"""FOVEATED LOCALIZATION diagnostic (the right fix for small-object localization): zoom the agentview to HI-RES so the
~20px object becomes ~80-200px, detect its precise pixel there, and map to world with the HI-RES camera matrix. Isolates
whether foveation gives a precise pixel -> precise xy (vs the 256px ~7cm wall). Handles the render/array FLIP by VERIFYING
the whole chain against the oracle HR projection (oracle pixel -> ray-plane should be ~0). Compares: low-res(256) vs
foveated(HR) localization, base-contact@table vs dt-peak@obj-height, for the TRUE target mask (identity isolated).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/diag_foveate_loc.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20 --hr 1024
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def ray_plane_px(sim, cam, r_native, c, z_plane, RES):
    """Pixel (r_native,c) in the camera-matrix-native frame at resolution RES -> world point on plane z=z_plane."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, RES, RES); inv = np.linalg.inv(w2p)
    rr = min(max(int(round(r_native)), 0), RES - 1); cc = min(max(int(round(c)), 0), RES - 1)
    d1 = np.full((RES, RES, 1), 1.0, np.float32); d2 = np.full((RES, RES, 1), 2.0, np.float32)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d2[None], inv)[0])
    ray = q2 - q1; t = (z_plane - q1[2]) / ray[2] if abs(ray[2]) > 1e-9 else 0.0
    return (q1 + t * ray).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--hr", type=int, default=1024)
    ap.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from scipy.ndimage import distance_transform_edt
    dev = "cuda"
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, min_mask_region_area=60)
    R, HR = args.res, args.hr; sc = HR / R
    # error accumulators: [resolution][center] -> list
    err = {"lo_base": [], "lo_dt": [], "hr_base": [], "hr_dt": [], "verify_lo": [], "verify_hr": []}
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T])
        if rb.get(T) is None: env.close(); continue
        obj = body_pos(sim, rb[T]).astype(np.float64); zt = obj[2]
        # oracle pixels (matrix-native) at lo and hi res, to (a) find the target mask and (b) VERIFY the flip chain
        w2p_lo = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        w2p_hr = cu.get_camera_transform_matrix(sim, "agentview", HR, HR)
        px_lo = cu.project_points_from_world_to_camera(obj[None], w2p_lo, R, R)[0]      # (row,col) native, lo
        px_hr = cu.project_points_from_world_to_camera(obj[None], w2p_hr, HR, HR)[0]    # native, hi
        # VERIFY: oracle pixel -> ray-plane@zt should return obj (chain/flip sanity, independent of detection)
        err["verify_lo"].append(np.linalg.norm(ray_plane_px(sim, "agentview", px_lo[0], px_lo[1], zt, R)[:2] - obj[:2]))
        err["verify_hr"].append(np.linalg.norm(ray_plane_px(sim, "agentview", px_hr[0], px_hr[1], zt, HR)[:2] - obj[:2]))
        # render upright images; rendered-row -> native-row = RES-1-rendered_row
        img_lo = np.asarray(obs["agentview_image"])[::-1].copy()                          # upright, lo
        img_hr = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()  # upright, hi
        # pick the TRUE target mask in each image = the mask whose centroid is nearest the oracle (upright) pixel
        def target_mask(img, oracle_native, RES):
            orow_up = RES - 1 - oracle_native[0]; ocol = oracle_native[1]; best = None; bd = 1e9
            for m in gen.generate(img):
                seg = m["segmentation"]; a = int(seg.sum())
                if a < 20 or a > 0.3 * RES * RES: continue
                ys, xs = np.where(seg); d = np.hypot(ys.mean() - orow_up, xs.mean() - ocol)
                if d < bd: bd = d; best = seg
            return best
        # WRIST camera (near-vertical -> localization should work; test foveated wrist too)
        pxw_lo = cu.project_points_from_world_to_camera(obj[None], cu.get_camera_transform_matrix(sim, "robot0_eye_in_hand", R, R), R, R)[0]
        pxw_hr = cu.project_points_from_world_to_camera(obj[None], cu.get_camera_transform_matrix(sim, "robot0_eye_in_hand", HR, HR), HR, HR)[0]
        wlo_in = (0 <= pxw_lo[0] < R) and (0 <= pxw_lo[1] < R)
        if wlo_in:
            wimg_lo = np.asarray(obs["robot0_eye_in_hand_image"])[::-1].copy()
            wimg_hr = np.asarray(sim.render(width=HR, height=HR, camera_name="robot0_eye_in_hand"))[::-1].copy()
            for tag, img, oracle_native, RES in [("wlo", wimg_lo, pxw_lo, R), ("whr", wimg_hr, pxw_hr, HR)]:
                seg = target_mask(img, oracle_native, RES)
                if seg is None: continue
                ys, xs = np.where(seg); dt = distance_transform_edt(seg); pk = np.unravel_index(int(np.argmax(dt)), dt.shape)
                r_nat = RES - 1 - pk[0]
                err.setdefault(f"{tag}_dt", []).append(np.linalg.norm(ray_plane_px(sim, "robot0_eye_in_hand", r_nat, pk[1], zt, RES)[:2] - obj[:2]))
        for tag, img, oracle_native, RES in [("lo", img_lo, px_lo, R), ("hr", img_hr, px_hr, HR)]:
            seg = target_mask(img, oracle_native, RES)
            if seg is None: continue
            ys, xs = np.where(seg)
            base_up = int(ys.max()); base_c = int(xs[ys >= ys.max() - 2].mean())          # base-contact (upright)
            r_native_base = RES - 1 - base_up
            err[f"{tag}_base"].append(np.linalg.norm(ray_plane_px(sim, "agentview", r_native_base, base_c, 0.0, RES)[:2] - obj[:2]))
            dt = distance_transform_edt(seg); pk = np.unravel_index(int(np.argmax(dt)), dt.shape)  # dt-peak (upright)
            r_native_dt = RES - 1 - pk[0]
            err[f"{tag}_dt"].append(np.linalg.norm(ray_plane_px(sim, "agentview", r_native_dt, pk[1], zt, RES)[:2] - obj[:2]))
        print(f"  {pathlib.Path(bf).stem[:24]:26s} lo_dt={np.mean(err['lo_dt'][-1:])*100 if err['lo_dt'] else 0:4.1f}cm  hr_dt={np.mean(err['hr_dt'][-1:])*100 if err['hr_dt'] else 0:4.1f}cm", flush=True)
        env.close()
    print(f"\n=== FOVEATED LOCALIZATION (agentview {R} vs {HR}), {pathlib.Path(args.bddl_dir).name} ===", flush=True)
    print(f"  VERIFY (oracle pixel->ray-plane, flip sanity):  lo={np.mean(err['verify_lo'])*100:.2f}cm  hr={np.mean(err['verify_hr'])*100:.2f}cm  (both should be ~0)", flush=True)
    for k in ["lo_base", "lo_dt", "hr_base", "hr_dt", "wlo_dt", "whr_dt"]:
        if err.get(k): print(f"  {k:8s} xy_err = {np.mean(err[k])*100:5.2f}cm   (n={len(err[k])})", flush=True)
    print("FOVLOC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
