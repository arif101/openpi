"""Stage 2 v3 binding source: SigLIP open-vocab localization (clean, un-captured).

For each scene/object/init: SigLIP heatmap for "a {object}" over agentview patches -> softmax centroid
+ argmax peak (normalized 2D) = where the named object is. Save 4-vec [cx,cy,px,py] + true 3D pos.
The grounding (which object) is SigLIP open-vocab (separated T vs M by ~11 patches); only the 2D->3D
planar map is learned. -> data/siglip/*.npz, train with distill_box --data data/siglip.
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--out", default="data/siglip")
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--k-init", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.02)
    args = ap.parse_args()
    import torch
    from transformers import SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(args.model).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def heat(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(model.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(model.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0]))
        s = (P @ e).reshape(g, g).float().cpu().numpy()
        ij = np.indices((g, g)).reshape(2, -1).T.astype(np.float32)   # (row,col) per patch
        w = np.exp((s.ravel() - s.max()) / args.tau); w /= w.sum() + 1e-8
        c = (w[:, None] * ij).sum(0) / (g - 1)                        # softmax centroid normalized
        pk = np.array(np.unravel_index(s.argmax(), s.shape), np.float32) / (g - 1)
        return np.concatenate([c, pk]).astype(np.float32)             # [cx,cy,px,py]

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
            img = np.asarray(obs["agentview_image"])[::-1]
            for o in present:
                oname = re.sub(r"_\d+$", "", o).replace("_", " ")
                np.savez(out / f"{pathlib.Path(bf).stem[:22]}_{o}_{k}.npz",
                         box=heat(img, f"a {oname}"), pos=body_pos(sim, rb[o]).astype(np.float32), obj=o)
                saved += 1
            env.close()
        print(f"  {pathlib.Path(bf).stem[:30]:32s} ({len(present)} objs)", flush=True)
    print(f"\nSAVED {saved} siglip->pos pairs -> {out}", flush=True)
    print("COLLECT_SIGLIP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
