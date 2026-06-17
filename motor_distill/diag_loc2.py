"""Pinpoint the localization failure: per swap object, compare (a) single-pixel depth unproject, (b) median-mask
depth, (c) ray->table-plane intersection (depth-free, position-independent). Report xy-error vs body_pos. Find the
robust method."""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="/root/LIBERO-Pro-data/bddl_files/libero_object_swap")
    ap.add_argument("--init-dir", default="/root/LIBERO-Pro-data/init_files/libero_object_swap")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()
    import torch
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    R = args.res
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True, camera_segmentations="instance")
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T]); pos = body_pos(sim, rb[T])
        depth = np.asarray(obs["agentview_depth"]); segN = np.asarray(obs["agentview_segmentation_instance"])
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R); inv = np.linalg.inv(w2p)
        real = cu.get_real_depth_map(sim, depth); real2 = np.asarray(real).reshape(R, R); seg = segN.reshape(R, R)
        px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
        r0 = int(min(max(px[0], 0), R - 1)); c0 = int(min(max(px[1], 0), R - 1))
        # (a) single pixel
        pa = cu.transform_from_pixels_to_world(np.array([[r0, c0]]), real[None], inv)[0]
        # (b) median-mask
        sid = seg[r0, c0]; ys, xs = np.where(seg == sid) if sid != 0 else (np.array([r0]), np.array([c0]))
        if len(ys) == 0: ys, xs = np.array([r0]), np.array([c0])
        dmed = float(np.median(real2[ys, xs])); keep = np.abs(real2[ys, xs] - dmed) < 0.02
        if keep.sum() == 0: keep = np.ones(len(ys), bool)
        rc, cc = int(round(ys[keep].mean())), int(round(xs[keep].mean()))
        pb = cu.transform_from_pixels_to_world(np.array([[rc, cc]]), real[None], inv)[0]
        # (c) ray -> table plane (z = pos[2] proxy for true; deploy uses a fixed table z)
        d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
        q1 = cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d1[None], inv)[0]
        q2 = cu.transform_from_pixels_to_world(np.array([[r0, c0]]), d2[None], inv)[0]
        ray = np.asarray(q2) - np.asarray(q1); zt = pos[2]
        tt = (zt - q1[2]) / ray[2] if abs(ray[2]) > 1e-6 else 0.0; pc = np.asarray(q1) + tt * ray
        e = lambda p: float(np.linalg.norm(np.asarray(p)[:2] - pos[:2]))   # XY error
        print(f"{stem[:24]:26s} px=({r0},{c0}) mask={len(ys):4d}  "
              f"a_pix={e(pa)*100:5.1f}cm  b_mask={e(pb)*100:5.1f}cm  c_rayplane={e(pc)*100:5.1f}cm", flush=True)
        env.close()
    print("DIAG2_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
