"""De-hardcoding via SAM: does SAM's class-agnostic mask proposer COVER the graspable objects (no body_pos, no
vocabulary)? Generate SAM masks on the agentview; for each object, check if a mask centroid lands near its true
projected center. SAM finds all object regions regardless of size -> if recall is high, SAM masks + DINOv2 identity
+ ray-plane = fully honest perception (replaces the privileged body_pos proposals that OWLv2 (42%) couldn't).
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="/root/LIBERO-Pro-data/bddl_files/libero_object_swap")
    ap.add_argument("--init-dir", default="/root/LIBERO-Pro-data/init_files/libero_object_swap")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth")
    args = ap.parse_args()
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda"
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=24, min_mask_region_area=40)
    R = args.res; hit = 0; tot = 0; errs = []; nmasks = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        graspables = [o for o in objs if "basket" not in o]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
        env.seed(args.init_start); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[args.init_start]) if inits is not None else env.reset()
        img = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])   # un-flip to upright
        masks = gen.generate(img); nmasks.append(len(masks))
        # mask centroids (in upright-image row,col), filter out huge (table/background) masks
        cents = []
        for m in masks:
            seg = m["segmentation"]; a = seg.sum()
            if a < 20 or a > 0.25 * R * R: continue
            ys, xs = np.where(seg); cents.append((ys.mean(), xs.mean()))
        rb = resolve_bodies(sim, graspables); w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        for o in graspables:
            if rb.get(o) is None: continue
            tot += 1
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            tr = R - 1 - px[0]; tc = px[1]   # account for the vertical flip
            if cents:
                d = min(np.hypot(c[0] - tr, c[1] - tc) for c in cents)
                if d < 18: hit += 1; errs.append(float(d))
        print(f"  {pathlib.Path(bf).stem[:26]:28s} masks={len(masks)} cands={len(cents)} graspables={len([o for o in graspables if rb.get(o)])}", flush=True)
        env.close()
    print(f"\n=== SAM recall: {hit}/{tot} = {100*hit/max(tot,1):.0f}%  centroid err(px) mean={np.mean(errs) if errs else -1:.1f}  avg masks={np.mean(nmasks):.0f} ===", flush=True)
    print("SAMREC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
