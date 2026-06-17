"""DEPTH-based localization (ActiveVLA-validated path): unproject the target mask to 3D points with PROPERLY LINEARIZED
depth -> 3D centroid = true object center, NO silhouette-offset (the geometric wall foveation couldn't break). Tests the
hypothesis that the old '16-47cm depth error' was a buffer-LINEARIZATION bug (median-mask on the RAW nonlinear OpenGL
buffer) and that robosuite's get_real_depth_map fixes it. Includes a VERIFY (unproject the true object's pixel -> should
land on the object) to pin the flip/convention. Compares depth-centroid vs the ~8cm silhouette ray-plane.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/diag_depth_loc.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    ap.add_argument("--flip", type=int, default=1)   # obs depth/image upright<->matrix-native row flip
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = "cuda"; R = args.res
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, min_mask_region_area=60)
    cen_err = []; med_err = []; sv_err = []; verify = []
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
        rb = resolve_bodies(sim, [T])
        if rb.get(T) is None: env.close(); continue
        obj = body_pos(sim, rb[T]).astype(np.float64)
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R); inv = np.linalg.inv(w2p)
        # PROPERLY LINEARIZED metric depth (the fix): get_real_depth_map converts the normalized OpenGL buffer via near/far
        draw = np.asarray(obs["agentview_depth"]).reshape(R, R, 1).astype(np.float32)
        real = cu.get_real_depth_map(sim, draw)                       # (R,R,1) metric depth
        if args.flip: real = real[::-1].copy()                        # obs depth is upright -> flip to matrix-native rows
        d1m = np.ones((1, R, R, 1), np.float32); d2m = 2 * np.ones((1, R, R, 1), np.float32)
        def unproject(pix):  # pix: (N,2) row,col (matrix-native) -> world (N,3). world(z) = q1 + (z-1)(q2-q1), z=real depth
            out = []
            for r, c in pix:
                r = int(min(max(r, 0), R - 1)); c = int(min(max(c, 0), R - 1))
                q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d1m, inv)[0])
                q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), d2m, inv)[0])
                z = float(real[r, c, 0]); out.append(q1 + (z - 1.0) * (q2 - q1))
            return np.asarray(out)
        # VERIFY: oracle object pixel -> unproject with real depth -> should land on/near the object surface
        px = cu.project_points_from_world_to_camera(obj[None], w2p, R, R)[0]
        r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
        pv = unproject(np.array([[r0, c0]]))[0]
        verify.append(np.linalg.norm(pv[:2] - obj[:2]))
        # honest: SAM -> target mask (oracle-nearest centroid, identity isolated) on the UPRIGHT image
        img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
        orow_up = R - 1 - px[0]; best = None; bd = 1e9
        for m in gen.generate(img_up):
            seg = m["segmentation"]; a = int(seg.sum())
            if a < 20 or a > 0.3 * R * R: continue
            ys, xs = np.where(seg); d = np.hypot(ys.mean() - orow_up, xs.mean() - px[1])
            if d < bd: bd = d; best = (ys, xs)
        if best is None: env.close(); continue
        ys_up, xs_up = best
        if len(ys_up) > 60:    # subsample mask for the per-pixel unprojection
            idx = np.random.RandomState(0).choice(len(ys_up), 60, replace=False); ys_up_s, xs_up_s = ys_up[idx], xs_up[idx]
        else: ys_up_s, xs_up_s = ys_up, xs_up
        rows_native = (R - 1 - ys_up_s) if args.flip else ys_up_s        # mask pixels -> matrix-native rows
        pix = np.stack([rows_native, xs_up_s], 1)
        pts = unproject(pix)   # (N,3) world points
        # depth-based outlier rejection: keep ONLY points ABOVE the table (object surface), drop mask-bleed onto table/bg
        onobj = pts[(pts[:, 2] > 0.005) & (pts[:, 2] < 0.25)]
        if len(onobj) < 3: onobj = pts
        cen_err.append(np.linalg.norm(onobj[:, :2].mean(0) - obj[:2]))      # filtered centroid xy
        med_err.append(np.linalg.norm(np.median(onobj[:, :2], 0) - obj[:2]))  # filtered median xy (robust)
        # silhouette ray-plane baseline (the ~8cm wall): mask base-contact @ table z=0
        base_up = int(ys_up.max()); base_c = int(xs_up[ys_up >= ys_up.max() - 2].mean()); rb_n = R - 1 - base_up
        d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
        q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rb_n, base_c]]), d1[None], inv)[0])
        q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rb_n, base_c]]), d2[None], inv)[0])
        ray = q2 - q1; t = (0.0 - q1[2]) / ray[2]; sp = q1 + t * ray
        sv_err.append(np.linalg.norm(sp[:2] - obj[:2]))
        print(f"  {pathlib.Path(bf).stem[:24]:26s} depth_centroid={cen_err[-1]*100:4.1f}cm  depth_median={med_err[-1]*100:4.1f}cm  verify={verify[-1]*100:4.1f}cm  silhouette={sv_err[-1]*100:4.1f}cm", flush=True)
        env.close()
    print(f"\n=== DEPTH LOCALIZATION (linearized), {pathlib.Path(args.bddl_dir).name} ===", flush=True)
    print(f"  VERIFY (oracle pixel + real depth -> object):  {np.mean(verify)*100:.2f}cm   (should be small if depth+flip OK)", flush=True)
    print(f"  DEPTH centroid xy = {np.mean(cen_err)*100:.2f}cm   |   DEPTH median xy = {np.mean(med_err)*100:.2f}cm   |   SILHOUETTE ray-plane = {np.mean(sv_err)*100:.2f}cm", flush=True)
    print("DEPTHLOC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
