"""Stage 2 v4: SigLIP clean 2D localization + DEPTH -> accurate 3D goal.

SigLIP heatmap peak (clean open-vocab, un-captured) gives the target pixel; the agentview DEPTH at
that pixel resolves the planar ambiguity. Save [u, v, depth] + true 3D; a tiny MLP [u,v,depth]->3D
learns the camera unprojection (absorbs intrinsics/flip). Should give ~cm goals at clean locations.
-> data/siglipd/*.npz ; train with distill_box --data data/siglipd.
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--out", default="data/siglipd")
    ap.add_argument("--n-scenes", type=int, default=10); ap.add_argument("--k-init", type=int, default=8)
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
    def peak(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(model.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(model.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0])); s = (P @ e).reshape(g, g).float().cpu().numpy()
        r, c = np.unravel_index(s.argmax(), s.shape)
        return r / (g - 1), c / (g - 1)                              # (v_norm, u_norm) in [0,1]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n_scenes]
    saved = 0; depth_ok = None
    for bf in bddls:
        _, objs, targets, distractors = parse_bddl(bf)
        present = list(dict.fromkeys((targets or []) + (distractors or [])))
        if not present:
            continue
        for k in range(args.k_init):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256,
                                     camera_depths=True)
            env.seed(100 + k); env.reset(); obs = env.reset()
            sim = env.env.sim; rb = resolve_bodies(sim, present)
            img = np.asarray(obs["agentview_image"]); H, W = img.shape[:2]
            depth = np.asarray(obs.get("agentview_depth"))
            if depth_ok is None:
                depth_ok = depth is not None and depth.size > 0
                print(f"  depth available={depth_ok} shape={None if depth is None else depth.shape}", flush=True)
            if not depth_ok:
                env.close(); break
            depth = depth.reshape(H, W)
            for o in present:
                oname = re.sub(r"_\d+$", "", o).replace("_", " ")
                v, u = peak(img, f"a {oname}")
                d = float(depth[min(int(v * (H - 1)), H - 1), min(int(u * (W - 1)), W - 1)])
                np.savez(out / f"{pathlib.Path(bf).stem[:22]}_{o}_{k}.npz",
                         box=np.array([u, v, d], np.float32), pos=body_pos(sim, rb[o]).astype(np.float32), obj=o)
                saved += 1
            env.close()
        if not depth_ok:
            break
        print(f"  {pathlib.Path(bf).stem[:30]:32s} ({len(present)} objs)", flush=True)
    print(f"\nSAVED {saved} siglip+depth->pos pairs -> {out}", flush=True)
    print("COLLECT_SIGLIPD_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
