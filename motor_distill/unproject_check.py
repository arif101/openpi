"""Foundation check for the mask+depth+geometry binder: verify robosuite pixel<->world unprojection,
then compare localizer sources (SigLIP region centroid vs GDINO box center) for language->3D accuracy.

For each captured object scene, for the COUNTERFACTUAL target T:
  geom round-trip: project true 3D -> pixel -> unproject(pixel, depth) -> 3D ; error should be ~0 (sanity).
  SigLIP region: heatmap for "a T" -> patches above thresh -> pixel centroid + MEDIAN real-depth -> unproject.
  GDINO box:     detect_box("a T") -> box center + median depth in box -> unproject (None if not detected).
Reports per-source 3D error vs true T, GDINO detection rate, and which flip convention is correct.
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--region-frac", type=float, default=0.85)
    args = ap.parse_args()
    import torch
    from transformers import SiglipModel, AutoProcessor
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from cf_harness import parse_bddl, resolve_bodies, body_pos, detect_box, clean_query
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sg = SiglipModel.from_pretrained(args.model).to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.model)
    H = W = 256

    @torch.no_grad()
    def heat(img, text):
        px = proc(images=img, return_tensors="pt").to(dev)
        P = torch.nn.functional.normalize(sg.vision_model(**px).last_hidden_state[0], dim=-1)
        tk = proc(text=[text], padding="max_length", return_tensors="pt").to(dev)
        e = torch.nn.functional.normalize(sg.text_model(**tk).pooler_output[0], dim=0)
        g = int(np.sqrt(P.shape[0]))
        return (P @ e).reshape(g, g).float().cpu().numpy(), g

    def unproj(px_xy, depth_map, c2w):
        # robosuite transform_from_pixels_to_world expects pixels as [...,2] in (row, col)=(y,x)
        return CU.transform_from_pixels_to_world(np.array(px_xy, np.int32), depth_map, c2w)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    eg, es, ed, ndet, nflip = [], [], [], 0, 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T]); tp = body_pos(sim, rb[T]).astype(np.float64)
        img = np.asarray(obs["agentview_image"]); depth = np.asarray(obs["agentview_depth"])
        rdepth = CU.get_real_depth_map(sim, depth.reshape(H, W, 1))
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        # geom round-trip: world->pixel then back. project returns (N,2)
        proj = CU.project_points_from_world_to_camera(tp[None], w2c, H, W)[0]   # (row,col)?
        r, c = int(np.clip(proj[0], 0, H - 1)), int(np.clip(proj[1], 0, W - 1))
        # try both flip conventions for the obs frame, pick the better via geom round-trip
        cand = {"raw": (r, c), "flip": (H - 1 - r, c)}
        errs = {k: float(np.linalg.norm(unproj([[rr, cc]], rdepth, c2w)[0] - tp))
                for k, (rr, cc) in cand.items()}
        conv = min(errs, key=errs.get); eg.append(errs[conv]); nflip += (conv == "flip")
        fr = (lambda rr: H - 1 - rr) if conv == "flip" else (lambda rr: rr)
        # SigLIP region centroid + median depth
        s, g = heat(img[::-1] if conv == "flip" else img, f"a {tn}")
        thr = args.region_frac * s.max() + (1 - args.region_frac) * s.mean()
        ys, xs = np.where(s >= thr)
        pys = (ys / (g - 1) * (H - 1)).astype(int); pxs = (xs / (g - 1) * (W - 1)).astype(int)
        cy, cx = int(np.median(pys)), int(np.median(pxs))
        dmed = float(np.median(rdepth[pys, pxs, 0]))
        sworld = CU.transform_from_pixels_to_world(np.array([[fr(cy) if False else cy, cx]]),
                                                   np.full((H, W, 1), dmed), c2w)[0]
        es.append(float(np.linalg.norm(sworld - tp)))
        # GDINO box
        box = detect_box(img, clean_query(T), dev)
        if box is not None:
            ndet += 1
            x0, y0, x1, y1 = box; bcx, bcy = (x0 + x1) // 2, (y0 + y1) // 2
            reg = rdepth[max(0, y0):y1, max(0, x0):x1, 0]
            bd = float(np.median(reg)) if reg.size else dmed
            bworld = CU.transform_from_pixels_to_world(np.array([[bcy, bcx]]), np.full((H, W, 1), bd), c2w)[0]
            ed.append(float(np.linalg.norm(bworld - tp)))
        env.close()
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={tn:13s} | geom={errs[conv]*100:5.1f}cm({conv}) "
              f"siglip={es[-1]*100:5.1f}cm gdino={'%.1f'%(ed[-1]*100) if box is not None else 'MISS':>6}", flush=True)
    f = lambda a: (np.mean(a) * 100 if a else float('nan'))
    print(f"\n=== UNPROJECTION CALIBRATION (N={len(eg)}) ===", flush=True)
    print(f"  geom round-trip err = {f(eg):.1f}cm  (flip-convention used {nflip}/{len(eg)})  <- must be ~0-2cm", flush=True)
    print(f"  SigLIP-region->3D   = {f(es):.1f}cm  (median over depth+region)", flush=True)
    print(f"  GDINO-box->3D       = {f(ed):.1f}cm  (detected {ndet}/{len(eg)})", flush=True)
    print("UNPROJECT_CHECK_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
