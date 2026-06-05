"""Pin vflip convention; isolate localizer quality: does GDINO-box-center / SigLIP-argmax-peak land on
the TARGET pixel, and what's the 3D error using REAL depth at that exact pixel? (geom verified 2.6cm)."""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--cam", default="agentview")
    args = ap.parse_args()
    import torch
    from transformers import SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, clean_query
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sg = SiglipModel.from_pretrained(args.model).to(dev).eval(); proc = AutoProcessor.from_pretrained(args.model)
    H = W = 256

    @torch.no_grad()
    def peak(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(sg.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(sg.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0])); s = (P @ e).reshape(g, g).float().cpu().numpy()
        r, c = np.unravel_index(s.argmax(), s.shape)
        return int(r / (g - 1) * (H - 1)), int(c / (g - 1) * (W - 1))

    def up(rc, dmap, c2w):
        return CU.transform_from_pixels_to_world(np.array(rc, float), dmap, c2w)[:3]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    gpix, spix, g3d, s3d, ndet = [], [], [], [], 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T]); tp = body_pos(sim, rb[T]).astype(np.float64)
        img = np.asarray(obs["agentview_image"])[::-1].copy()           # vflip frame
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        proj = np.asarray(CU.project_points_from_world_to_camera(tp[None], w2c, H, W)[0])
        tr, tc = int(np.clip(proj[0], 0, H - 1)), int(np.clip(proj[1], 0, W - 1))
        tr_f = H - 1 - tr                                                # true pixel in vflip frame
        # SigLIP peak
        sr, sc = peak(img, f"a {tn}")
        spix.append(np.hypot(sr - tr_f, sc - tc)); s3d.append(float(np.linalg.norm(up((sr, sc), rd, c2w) - tp)))
        # GDINO box center
        box = detect_box(img, clean_query(T), dev)
        if box is not None:
            ndet += 1; x0, y0, x1, y1 = box; bcy, bcx = (y0 + y1) // 2, (x0 + x1) // 2
            gpix.append(np.hypot(bcy - tr_f, bcx - tc)); g3d.append(float(np.linalg.norm(up((bcy, bcx), rd, c2w) - tp)))
            gp = f"{gpix[-1]:.0f}px/{g3d[-1]*100:.0f}cm"
        else:
            gp = "MISS"
        env.close()
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={tn:13s} truepix=({tr_f},{tc}) | "
              f"siglip {spix[-1]:.0f}px/{s3d[-1]*100:.0f}cm | gdino {gp}", flush=True)
    f = lambda a: (float(np.mean(a)) if a else float('nan'))
    print(f"\n=== LOCALIZER (N={len(s3d)}, vflip pinned) ===", flush=True)
    print(f"  SigLIP-peak: {f(spix):.0f}px to true,  {f(s3d)*100:.0f}cm 3D", flush=True)
    print(f"  GDINO-box:   {f(gpix):.0f}px to true,  {f(g3d)*100:.0f}cm 3D  (det {ndet}/{len(s3d)})", flush=True)
    print("LOC_CHECK_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
