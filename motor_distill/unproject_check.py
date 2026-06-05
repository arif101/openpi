"""Calibrate robosuite pixel<->world unprojection (auto-resolve flip/axis convention via round-trip),
then compare SigLIP-region vs GDINO-box as the language->3D localizer source.

transform_from_pixels_to_world: pixels shape (2,)=(row,col), full depth_map (H,W,1), c2w 4x4.
"""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--region-frac", type=float, default=0.9)
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

    def up(rc, dmap, c2w):
        return CU.transform_from_pixels_to_world(np.array(rc, float), dmap, c2w)[:3]

    # candidate conventions: (flip depth vertically?, pixel order from projection)
    CONV = [("vraw", lambda d: d, lambda p: (p[0], p[1])),
            ("vflip", lambda d: d[::-1].copy(), lambda p: (p[0], p[1])),
            ("vraw_swap", lambda d: d, lambda p: (p[1], p[0])),
            ("vflip_swap", lambda d: d[::-1].copy(), lambda p: (p[1], p[0]))]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    eg, es, ed, ndet = [], [], [], 0
    conv_votes = {}
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T]); tp = body_pos(sim, rb[T]).astype(np.float64)
        img = np.asarray(obs["agentview_image"]); depth = np.asarray(obs["agentview_depth"]).reshape(H, W, 1)
        rdepth = CU.get_real_depth_map(sim, depth)
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        proj = np.asarray(CU.project_points_from_world_to_camera(tp[None], w2c, H, W)[0])
        # pick the convention with lowest round-trip error for THIS scene
        best = None
        for name, dflip, order in CONV:
            rc = order(proj); rc = (int(np.clip(rc[0], 0, H - 1)), int(np.clip(rc[1], 0, W - 1)))
            err = float(np.linalg.norm(up(rc, dflip(rdepth), c2w) - tp))
            if best is None or err < best[1]:
                best = (name, err, dflip, order)
        cname, gerr, dflip, order = best
        conv_votes[cname] = conv_votes.get(cname, 0) + 1
        eg.append(gerr)
        rd = dflip(rdepth)
        simg = img[::-1].copy() if "vflip" in cname else img
        # SigLIP region centroid + median depth over region
        s, g = heat(simg, f"a {tn}")
        thr = args.region_frac * s.max() + (1 - args.region_frac) * s.mean()
        ys, xs = np.where(s >= thr)
        pys = np.clip((ys / (g - 1) * (H - 1)).astype(int), 0, H - 1)
        pxs = np.clip((xs / (g - 1) * (W - 1)).astype(int), 0, W - 1)
        cy, cx = int(np.median(pys)), int(np.median(pxs))
        # build a depth map masked to region-median so bilinear at (cy,cx) ~ region median
        dmed = float(np.median(rd[pys, pxs, 0]))
        sw = up((cy, cx), np.full((H, W, 1), dmed), c2w)
        es.append(float(np.linalg.norm(sw - tp)))
        # GDINO box
        box = detect_box(simg, clean_query(T), dev)
        if box is not None:
            ndet += 1
            x0, y0, x1, y1 = box; reg = rd[max(0, y0):y1, max(0, x0):x1, 0]
            bd = float(np.median(reg)) if reg.size else dmed
            bw = up(((y0 + y1) // 2, (x0 + x1) // 2), np.full((H, W, 1), bd), c2w)
            ed.append(float(np.linalg.norm(bw - tp)))
        env.close()
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={tn:13s} | geom={gerr*100:4.1f}cm[{cname}] "
              f"siglip={es[-1]*100:5.1f}cm gdino={'%.1f'%(ed[-1]*100) if box is not None else 'MISS':>6}", flush=True)
    f = lambda a: (float(np.mean(a)) * 100 if a else float('nan'))
    print(f"\n=== UNPROJECTION CALIBRATION (N={len(eg)}) ===", flush=True)
    print(f"  convention votes: {conv_votes}", flush=True)
    print(f"  geom round-trip   = {f(eg):.1f}cm   <- must be ~0-3cm for geometry to be correct", flush=True)
    print(f"  SigLIP-region->3D = {f(es):.1f}cm", flush=True)
    print(f"  GDINO-box->3D     = {f(ed):.1f}cm  (detected {ndet}/{len(eg)})", flush=True)
    print("UNPROJECT_CHECK_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
