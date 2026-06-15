"""SANITY CHECK (step 1a) for the general masked-depth node localizer.

Question: does masked/windowed depth recover each object's 3D position to a FEW CM -- for ALL
objects, including ones FAR from image center and TALL/short ones (the historical 19-60 cm
failure cases)? If yes, the localizer is GENERAL (not the tabletop-only ray_plane) and the
scene-graph's coarse-3D input is trustworthy -> we can build the auto-labeler + grounding head on it.

Isolation: we feed the ORACLE center pixel (projected from the ground-truth body) so this tests the
LOCALIZER ("given correct attention, is masked depth accurate?"), NOT the detector. Detector
accuracy is a separate test downstream.

Pass criteria (printed at the end):
  - median 3D error < ~3 cm, 90th-pct < ~5 cm
  - NO object > 10 cm (i.e. the old flip/parallax catastrophes are gone)
  - error roughly FLAT vs eccentricity (the flip bug made error grow with distance-from-center)

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python \
  motor_distill/diag_node_localizer.py --bddl-dir <object-suite bddls> --init-dir <inits> \
  --n 10 --trials 3 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from node_localizer import flipped_depth, localize


def obj_pixel(sim, body, R, cam="agentview"):
    """Oracle center pixel: project the ground-truth body position into the camera (test-only)."""
    import robosuite.utils.camera_utils as cu
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    return cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, R, R)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--container", default="basket")
    p.add_argument("--win", type=int, default=12); p.add_argument("--z-mode", default="surface")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv

    rows = []  # per (object, init): err_cm, exy_cm, ez_cm, eccentricity_px, gt_z
    worst = []
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"NODE-LOCALIZER sanity check on {pathlib.Path(args.bddl_dir).name} "
          f"({len(bddls)} tasks, win={args.win}, res={args.res})", flush=True)
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        graspables = [o for o in objs if args.container not in o]
        stem = pathlib.Path(bf).stem
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res,
                                     camera_widths=args.res, camera_depths=True)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables)
            dm = flipped_depth(sim, obs["agentview_depth"])
            ctr = args.res / 2.0
            for o in graspables:
                if rb.get(o) is None:
                    continue
                gt = body_pos(sim, rb[o]).astype(np.float32)
                px = obj_pixel(sim, rb[o], args.res)
                est = localize(sim, px, dm, args.res, win=args.win, z_mode=args.z_mode)
                if est is None:
                    continue
                e = est - gt
                ecm = float(np.linalg.norm(e) * 100); exy = float(np.linalg.norm(e[:2]) * 100)
                ez = float(abs(e[2]) * 100); ecc = float(np.linalg.norm(np.array(px[:2]) - ctr))
                rows.append((ecm, exy, ez, ecc, float(gt[2])))
                worst.append((ecm, o, stem))
            env.close()
        a = np.array([r[0] for r in rows]) if rows else np.array([0.0])
        print(f"  {stem[:30]:32s} running median={np.median(a):.2f}cm  n={len(rows)}", flush=True)

    if not rows:
        print("NO measurements -- check bddl/init dirs.", flush=True); return
    R = np.array(rows)  # cols: err, exy, ez, ecc, z
    err = R[:, 0]
    # eccentricity correlation: split near vs far from center
    med_ecc = np.median(R[:, 3]); near = err[R[:, 3] <= med_ecc]; far = err[R[:, 3] > med_ecc]
    corr = float(np.corrcoef(R[:, 3], err)[0, 1]) if len(err) > 2 else float("nan")
    print("\n================ NODE-LOCALIZER VERDICT ================", flush=True)
    print(f"N={len(err)} object-measurements", flush=True)
    print(f"3D error cm:  median={np.median(err):.2f}  mean={err.mean():.2f}  "
          f"p90={np.percentile(err,90):.2f}  max={err.max():.2f}", flush=True)
    print(f"  xy error:  median={np.median(R[:,1]):.2f}cm   z error: median={np.median(R[:,2]):.2f}cm", flush=True)
    print(f"  within 3cm: {100*np.mean(err<3):.0f}%   within 5cm: {100*np.mean(err<5):.0f}%   "
          f">10cm: {100*np.mean(err>10):.0f}%", flush=True)
    print(f"  eccentricity: near-center median={np.median(near):.2f}cm  far median={np.median(far):.2f}cm  "
          f"corr(ecc,err)={corr:+.2f}  (flat ~0 = no flip bug)", flush=True)
    worst.sort(reverse=True)
    print("  worst 5:", [(f"{e:.1f}cm", o[:14], s[:16]) for e, o, s in worst[:5]], flush=True)
    ok = (np.median(err) < 3.0 and np.percentile(err, 90) < 5.0 and err.max() < 10.0)
    print(f"\nVERDICT: {'PASS -> localizer is general; build the auto-labeler on it' if ok else 'FAIL -> diagnose before building (z-mode? win? convention?)'}", flush=True)
    print("NODELOC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
