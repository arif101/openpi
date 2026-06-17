"""Visual debugger for spatial relation binding: render the scene, overlay every SAM detection with its classified
label (+DINOv2 score), mark the chosen bowl (RED), GT target bowl (GREEN), anchors (CYAN). Shows WHY binding fails.
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from dino_separability import dino_feat
from eval_e2e_spatial import cls_of, parse_relation, target_cls_from_stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--init-start", type=int, default=20); ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--bind-res", type=int, default=1024); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ckpt", default="/root/openpi/sam_vit_b_01ec64.pth"); ap.add_argument("--proto-inits", default="30,32,34")
    ap.add_argument("--out", default="/root/openpi/logs/viz_spatial")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from PIL import Image, ImageDraw
    dev = "cuda"; R = args.res; HR = args.bind_res; sc = HR / R
    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(dev).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=25)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    # per-class bank (same as eval_e2e_spatial)
    acc = {}
    pinits = [int(x) for x in args.proto_inits.split(",")]
    for bf in bddls[:4]:
        instr, objs, targets, distractors = parse_bddl(bf)
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        for ti in pinits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R); env.seed(ti); env.reset()
            obs = env.set_init_state(inits[ti]); sim = env.env.sim
            hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            rb = resolve_bodies(sim, list(dict.fromkeys(objs)))
            for o in list(dict.fromkeys(objs)):
                if rb.get(o) is None: continue
                px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
                hy, hx = int((R - 1 - px[0]) * sc), int(px[1] * sc); s = 70
                cr = hi[max(0, hy - s):hy + s, max(0, hx - s):hx + s]
                if cr.size < 100: continue
                acc.setdefault(cls_of(o), []).append(dino_feat(cr, dev))
            env.close()
    bank = {k: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for k, v in acc.items() if v}
    print("classes:", sorted(bank), flush=True)
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    for bf in bddls[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        tcls = target_cls_from_stem(stem) or cls_of(targets[0])
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
        env.seed(args.init_start); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[args.init_start])
        img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
        hi = np.asarray(sim.render(width=HR, height=HR, camera_name="agentview"))[::-1].copy()
        pil = Image.fromarray(hi); dr = ImageDraw.Draw(pil)
        for m in gen.generate(img_up):
            seg = m["segmentation"]; a = int(seg.sum())
            if a < 20 or a > 0.3 * R * R: continue
            ys, xs = np.where(seg)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            pad = int(0.25 * max(y1 - y0, x1 - x0)) + 3
            cr = hi[int(max(0, y0 - pad) * sc):int(min(R, y1 + pad) * sc), int(max(0, x0 - pad) * sc):int(min(R, x1 + pad) * sc)]
            if cr.size < 100: continue
            f = dino_feat(cr, dev)
            scores = {k: float(f @ v) for k, v in bank.items()}
            k = max(scores, key=scores.get)
            col = (255, 60, 60) if k == "black_bowl" or k.endswith(tcls) or tcls.endswith(k) else (255, 220, 0)
            dr.rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=col, width=3)
            dr.text((x0 * sc, max(0, y0 * sc - 14)), f"{k[:14]} {scores[k]:.2f}", fill=col)
        # GT bowls + anchors
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        rb = resolve_bodies(sim, list(dict.fromkeys(objs)) + list(targets))
        for o in set(list(objs) + list(targets)):
            if rb.get(o) is None: continue
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            cy, cx = (R - 1 - px[0]) * sc, px[1] * sc
            is_gt = o in targets and (cls_of(o).endswith(tcls) or tcls.endswith(cls_of(o)))
            col = (0, 255, 0) if is_gt else (0, 220, 255)
            dr.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=col, width=4)
            dr.text((cx + 10, cy - 8), o, fill=col)
        outp = f"{args.out}/{stem[:48]}.png"; pil.save(outp)
        print(f"saved {outp}", flush=True)
        env.close()
    print("VIZSP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
