"""Verify object projections: render the full agentview, draw a big numbered marker at each object's body_pos projection
with a legend, so we can see if the projection lands on the right object (or if it's a flip/projection bug). Also dumps
each object's tight crop for inspection.
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--render", type=int, default=768)
    ap.add_argument("--target", default="ketchup"); ap.add_argument("--container", default="basket")
    ap.add_argument("--out", default="/root/openpi/logs/verify_proj.png")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from PIL import Image, ImageDraw
    R = args.render
    bf = next(b for b in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl"))) if args.target in b)
    instr, objs, targets, distractors = parse_bddl(bf); T = targets[0]
    inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init", weights_only=False))
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
    env.seed(args.init_start); env.reset(); sim = env.env.sim; env.set_init_state(inits[args.init_start])
    everything = [o for o in objs if args.container not in o] + [args.container + "_1"]
    rb = resolve_bodies(sim, everything); w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
    img_up = np.asarray(sim.render(width=R, height=R, camera_name="agentview"))[::-1].copy()
    pil = Image.fromarray(img_up); dr = ImageDraw.Draw(pil)
    print(f"scene: {pathlib.Path(bf).stem}  target={T}", flush=True)
    legend = []
    for i, o in enumerate(everything):
        if rb.get(o) is None: print(f"  [{i}] {o}: NO BODY", flush=True); continue
        wp = body_pos(sim, rb[o]); px = cu.project_points_from_world_to_camera(wp[None], w2p, R, R)[0]
        cy_up = R - 1 - px[0]; cx = px[1]; col = (0, 255, 0) if o == T else (255, 80, 80)
        dr.ellipse([cx - 10, cy_up - 10, cx + 10, cy_up + 10], outline=col, width=4)
        dr.text((cx + 12, cy_up - 6), str(i), fill=col)
        legend.append(f"[{i}] {o.rsplit('_',1)[0]}{' (TARGET)' if o==T else ''}  world=({wp[0]:.2f},{wp[1]:.2f},{wp[2]:.2f}) px=(row{px[0]:.0f},col{px[1]:.0f})")
        print("  " + legend[-1], flush=True)
    pil.save(args.out); print(f"saved {args.out}", flush=True); print("VP_EXIT=0", flush=True); env.close()


if __name__ == "__main__":
    main()
