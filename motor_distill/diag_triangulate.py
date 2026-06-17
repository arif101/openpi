"""FOUNDATIONAL test for two-view triangulation (the research-recommended honest localizer): intersect the agentview ray
and the wrist-at-start ray for the target -> 3D xy WITHOUT the single-view height ambiguity. Isolates the GEOMETRY: takes
the true object pixel in BOTH cameras, injects controlled pixel noise (honest detectors are off by a few px), triangulates
(closest point of the two skew rays), and reports xy error vs single-view ray-plane (~7cm). Also checks: does the
wrist-at-HOME actually see the tabletop objects (in-FOV)?  If triangulation < single-view at realistic noise -> build it.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/diag_triangulate.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def pixel_ray(sim, cam, r, c, R):
    """World-space ray (origin, unit dir) through pixel (r,c) of camera `cam`."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); inv = np.linalg.inv(w2p)
    d1 = np.full((R, R, 1), 1.0, np.float32); d2 = np.full((R, R, 1), 2.0, np.float32)
    rr = min(max(int(round(r)), 0), R - 1); cc = min(max(int(round(c)), 0), R - 1)
    q1 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d1[None], inv)[0])
    q2 = np.asarray(cu.transform_from_pixels_to_world(np.array([[rr, cc]]), d2[None], inv)[0])
    o = q1; d = q2 - q1; d = d / (np.linalg.norm(d) + 1e-9)
    return o.astype(np.float64), d.astype(np.float64)


def closest_point(o1, d1, o2, d2):
    """Midpoint of the shortest segment between two skew rays (least-squares 3D intersection)."""
    w0 = o1 - o2; a = d1 @ d1; b = d1 @ d2; c = d2 @ d2; dd = d1 @ w0; e = d2 @ w0
    denom = a * c - b * b
    if abs(denom) < 1e-9: return (o1 + o2) / 2
    s = (b * e - c * dd) / denom; t = (a * e - b * dd) / denom
    return (o1 + s * d1 + o2 + t * d2) / 2


def proj(sim, cam, world, R):
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(np.asarray(world)[None], w2p, R, R)[0]
    infov = (0 <= px[0] < R) and (0 <= px[1] < R)
    return px, infov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--wcam", default="robot0_eye_in_hand")
    ap.add_argument("--noise-px", default="0,1,2,3"); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    R = args.res; rng = np.random.RandomState(args.seed)
    noises = [float(x) for x in args.noise_px.split(",")]
    in_fov_w = []; tri = {nz: [] for nz in noises}; sv = {nz: [] for nz in noises}; triz = {nz: [] for nz in noises}
    wsv = {nz: [] for nz in noises}  # WRIST single-view: ray ∩ plane at the object's (calibrated) height z_obj
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(ti); env.reset(); sim = env.env.sim
        env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T])
        if rb.get(T) is None: env.close(); continue
        obj = body_pos(sim, rb[T]).astype(np.float64)
        pa, fa = proj(sim, "agentview", obj, R); pw, fw = proj(sim, args.wcam, obj, R)
        in_fov_w.append(int(fw))
        if not (fa and fw):
            print(f"  {pathlib.Path(bf).stem[:26]:28s} wrist_in_fov={fw}  (agentview_in_fov={fa}) -- SKIP triangulation", flush=True)
            env.close(); continue
        for nz in noises:
            for _ in range(args.reps):
                na = pa + rng.randn(2) * nz; nw = pw + rng.randn(2) * nz
                oa, da = pixel_ray(sim, "agentview", na[0], na[1], R)
                ow, dw = pixel_ray(sim, args.wcam, nw[0], nw[1], R)
                p3 = closest_point(oa, da, ow, dw)
                tri[nz].append(np.linalg.norm(p3[:2] - obj[:2])); triz[nz].append(abs(p3[2] - obj[2]))
                # single-view agentview ray-plane at table z=0 (the ~7cm baseline) with the SAME noisy pixel
                t = (0.0 - oa[2]) / da[2]; sp = oa + t * da
                sv[nz].append(np.linalg.norm(sp[:2] - obj[:2]))
                # WRIST single-view: ray ∩ plane at the object's calibrated height (wrist is near-overhead -> near-vertical
                # ray -> xy barely depends on z). No cross-view correspondence needed.
                if abs(dw[2]) > 0.1:
                    tw = (obj[2] - ow[2]) / dw[2]; wp = ow + tw * dw
                    wsv[nz].append(np.linalg.norm(wp[:2] - obj[:2]))
        print(f"  {pathlib.Path(bf).stem[:26]:28s} wrist_in_fov={fw}  tri@2px={np.mean(tri[2.0][-args.reps:])*100:4.1f}cm  sv@2px={np.mean(sv[2.0][-args.reps:])*100:4.1f}cm", flush=True)
        env.close()
    print(f"\n=== TWO-VIEW TRIANGULATION (agentview + wrist@start), {pathlib.Path(args.bddl_dir).name} ===", flush=True)
    print(f"  wrist-at-home sees target: {int(np.mean(in_fov_w)*100)}% ({sum(in_fov_w)}/{len(in_fov_w)})", flush=True)
    for nz in noises:
        if tri[nz]:
            print(f"  noise={nz:.0f}px  TRI xy={np.mean(tri[nz])*100:5.2f}cm  TRI z={np.mean(triz[nz])*100:5.2f}cm  |  AGENTVIEW-SV xy={np.mean(sv[nz])*100:5.2f}cm  |  WRIST-SV xy={np.mean(wsv[nz])*100:5.2f}cm", flush=True)
    print("TRIANG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
