"""Visualize GroundingDINO's prediction vs ground truth: render the agentview, draw GDINO's predicted box (RED) for the
target phrase, mark the TRUE target (GREEN dot) and distractors (YELLOW dots) with labels. Lets a human judge whether the
sim assets are even distinguishable at this resolution.
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--render", type=int, default=512)
    ap.add_argument("--out", default="/root/openpi/logs/gdino_viz"); ap.add_argument("--container", default="basket")
    ap.add_argument("--names", default="ketchup,bbq_sauce,orange_juice,tomato_sauce")  # which scenes to viz (by target)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image, ImageDraw
    dev = "cuda"; R = args.render
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    want = args.names.split(","); saved = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl"))):
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        if not any(w in T for w in want): continue
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(ti); env.reset(); sim = env.env.sim
        env.set_init_state(inits[ti]) if inits is not None else env.reset()
        graspables = [o for o in objs if args.container not in o]
        rb = resolve_bodies(sim, graspables + [args.container + "_1"])
        img_up = np.asarray(sim.render(width=R, height=R, camera_name="agentview"))[::-1].copy()
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        phrase = T.rsplit("_", 1)[0].replace("_", " ")
        # GroundingDINO box for the target phrase
        pil = Image.fromarray(img_up); inp = proc(images=pil, text=phrase + ".", return_tensors="pt").to(dev)
        with torch.no_grad(): out = gd(**inp)
        res = proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=0.25, text_threshold=0.20,
                                                          target_sizes=[pil.size[::-1]])[0]
        dr = ImageDraw.Draw(pil)
        # draw ALL GDINO boxes (red), label score
        for b, s in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()):
            dr.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=3)
            dr.text((b[0], max(0, b[1] - 12)), f"GDINO:{phrase} {s:.2f}", fill=(255, 0, 0))
        # draw true object positions: target GREEN, distractors YELLOW (projected; row is upright = R-1-native_row)
        for o in graspables:
            if rb.get(o) is None: continue
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            cy_up = R - 1 - px[0]; cx = px[1]; col = (0, 255, 0) if o == T else (255, 220, 0)
            dr.ellipse([cx - 5, cy_up - 5, cx + 5, cy_up + 5], outline=col, width=3)
            dr.text((cx + 6, cy_up - 6), o.rsplit("_", 1)[0], fill=col)
        outp = f"{args.out}/{phrase.replace(' ', '_')}.png"; pil.save(outp); saved.append(outp)
        print(f"saved {outp}  (target='{phrase}' GREEN, distractors YELLOW, GDINO RED, {len(res['boxes'])} boxes)", flush=True)
        env.close()
    print("VIZ_EXIT=0 " + " ".join(saved), flush=True)


if __name__ == "__main__":
    main()
