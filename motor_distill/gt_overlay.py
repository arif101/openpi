"""Ground-truth overlay: project EVERY object's body_pos onto the hi-res scene with labels. Reveals whether
body_pos matches the visible objects (i.e. whether the error-metric GT is trustworthy)."""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from PIL import Image, ImageDraw


def main():
    import re
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--task", default="cream_cheese"); p.add_argument("--res", type=int, default=1024)
    p.add_argument("--init", type=int, default=20); p.add_argument("--out", default="viz")
    args = p.parse_args()
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    R = args.res
    bf = [b for b in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl"))) if args.task in b][0]
    instr, objs, targets, distractors = parse_bddl(bf); T = targets[0]
    inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init", weights_only=False))
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
    env.seed(args.init); env.reset(); obs = env.set_init_state(inits[args.init]); sim = env.env.sim
    rgb = np.asarray(obs["agentview_image"]); up = rgb[::-1].copy()
    w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
    rb = resolve_bodies(sim, objs)
    im = Image.fromarray(up); d = ImageDraw.Draw(im)
    print(f"task={T}  objects:", flush=True)
    for o in objs:
        b = rb.get(o)
        if b is None: continue
        pos = body_pos(sim, b)
        px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
        r_up = R - 1 - px[0]; c = px[1]
        col = (0, 255, 0) if o == T else (255, 120, 0)
        d.ellipse([c - 10, r_up - 10, c + 10, r_up + 10], outline=col, width=4)
        d.text((c + 12, r_up - 6), re.sub(r"_\d+$", "", o), fill=col)
        print(f"  {o:18s} pos={np.round(pos,3)} -> upright[row,col]=[{int(r_up)},{int(c)}] {'<TARGET>' if o==T else ''}", flush=True)
    im.save(pathlib.Path(args.out) / f"{T}_ALLGT.png")
    env.close()
    print("GTOVERLAY_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
