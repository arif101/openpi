"""Diagnose the depth-unprojection pixel convention. For swap (off-center) objects, compute the surface point under
several [row,col] conventions and report error vs body_pos. The correct convention should give ~2-4cm (surface offset
from center), not ~21cm. Fixes the e2e localization bug.
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="/root/LIBERO-Pro-data/bddl_files/libero_object_swap")
    ap.add_argument("--init-dir", default="/root/LIBERO-Pro-data/init_files/libero_object_swap")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()
    import torch
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    R = args.res
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T]); pos = body_pos(sim, rb[T])
        depth = np.asarray(obs["agentview_depth"])
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
        real = cu.get_real_depth_map(sim, depth); inv = np.linalg.inv(w2p)
        a = int(min(max(px[0], 0), R - 1)); b = int(min(max(px[1], 0), R - 1))
        out = {}
        for name, (r, c) in {"cur[px0,px1]": (a, b), "swap[px1,px0]": (b, a)}.items():
            try:
                pt = cu.transform_from_pixels_to_world(np.array([[r, c]]), real[None], inv)[0]
                out[name] = float(np.linalg.norm(np.asarray(pt) - pos))
            except Exception as e:
                out[name] = f"ERR {e}"
        print(f"{stem[:26]:28s} px=({px[0]:.0f},{px[1]:.0f})  " + "  ".join(f"{k}={v:.3f}m" if isinstance(v,float) else f"{k}={v}" for k,v in out.items()), flush=True)
        env.close()
    print("DIAG_DEPTH_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
