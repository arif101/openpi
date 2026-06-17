"""Crop each object out of a HI-RES render, enlarge, and label it -> a grid so a human can judge whether the sim
groceries are even distinguishable (is 'ketchup' tellable from 'tomato sauce'?). TARGET labeled green, distractors gray.
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--hr", type=int, default=1024)
    ap.add_argument("--target", default="ketchup"); ap.add_argument("--container", default="basket")
    ap.add_argument("--out", default="/root/openpi/logs/crops.png"); ap.add_argument("--crop", type=int, default=110); ap.add_argument("--cell", type=int, default=224)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from PIL import Image, ImageDraw
    R = args.hr
    bf = next(b for b in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl"))) if args.target in b)
    instr, objs, targets, distractors = parse_bddl(bf); T = targets[0]
    inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init", weights_only=False))
    env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
    env.seed(args.init_start); env.reset(); sim = env.env.sim; env.set_init_state(inits[args.init_start])
    graspables = [o for o in objs if args.container not in o]
    rb = resolve_bodies(sim, graspables); w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
    img_up = np.asarray(sim.render(width=R, height=R, camera_name="agentview"))[::-1].copy()
    items = []
    for o in graspables:
        if rb.get(o) is None: continue
        px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
        cy_up = int(R - 1 - px[0]); cx = int(px[1]); h = args.crop
        crop = img_up[max(0, cy_up - h):cy_up + h, max(0, cx - h):cx + h]
        if crop.size < 100: continue
        im = Image.fromarray(crop).resize((args.cell, args.cell), Image.NEAREST)
        items.append((o.rsplit("_", 1)[0], o == T, im))
    # also a "what the policy sees" 256-native crop of the target for scale reference
    cols = len(items); cell = args.cell; pad = 28
    canvas = Image.new("RGB", (cols * cell, cell + pad), (30, 30, 30)); dr = ImageDraw.Draw(canvas)
    for i, (name, is_t, im) in enumerate(items):
        canvas.paste(im, (i * cell, pad))
        dr.text((i * cell + 4, 8), ("TARGET " if is_t else "") + name, fill=((0, 255, 0) if is_t else (200, 200, 200)))
        if is_t: dr.rectangle([i * cell, pad, (i + 1) * cell - 1, pad + cell - 1], outline=(0, 255, 0), width=4)
    canvas.save(args.out)
    print(f"saved {args.out}  ({cols} objects, target='{T.rsplit('_',1)[0]}', each cropped from {R}px render & enlarged)", flush=True)
    print("CROPS_EXIT=0", flush=True); env.close()


if __name__ == "__main__":
    main()
