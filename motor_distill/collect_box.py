"""Stage 2 v2 binding source: open-vocab DETECTOR box (not pi0.5's captured features).

For each scene/object/init: GroundingDINO box for "a {object}" on agentview + true 3D pos.
The detector grounds WHICH object open-vocab (not captured); only the box->3D planar localization
is learned (objects on a table = planar homography, accurate). Saves data/box/*.npz (box4, pos3).
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, clean_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--out", default="data/box")
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--k-init", type=int, default=8)
    args = ap.parse_args()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    saved = 0
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        present = list(dict.fromkeys((targets or []) + (distractors or [])))
        if not present:
            continue
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
            env.seed(100 + k); env.reset(); obs = env.reset()
            sim = env.env.sim; rb = resolve_bodies(sim, present)
            img = np.asarray(obs["agentview_image"])
            for o in present:
                box = detect_box(img, clean_query(o), dev)
                if box is None:
                    continue
                np.savez(out / f"{pathlib.Path(bf).stem[:22]}_{o}_{k}.npz",
                         box=np.array(box, np.float32) / 256.0, pos=body_pos(sim, rb[o]).astype(np.float32),
                         obj=o)
                saved += 1
            env.close()
        print(f"  {pathlib.Path(bf).stem[:30]:32s} ({len(present)} objs)", flush=True)
    print(f"\nSAVED {saved} box->pos pairs -> {out}", flush=True)
    print("COLLECT_BOX_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
