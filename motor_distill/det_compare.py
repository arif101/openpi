"""Decisive binding test: can a STRONG open-vocab detector localize these small tabletop objects?
Compare OWLv2-base and GroundingDINO-base (vs tiny) on the counterfactual target: pixel-distance to
true + 3D error (real depth + verified geometry, vflip frame). If a strong detector hits <~10px/<5cm,
binding is solved (open-vocab => generalizes by construction; depth => accurate)."""
from __future__ import annotations

import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--cam", default="agentview")
    args = ap.parse_args()
    import torch
    from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                              AutoProcessor, AutoModelForZeroShotObjectDetection)
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 256
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
    from PIL import Image

    @torch.no_grad()
    def owl(img, text):
        inp = owlp(text=[[text]], images=Image.fromarray(img), return_tensors="pt").to(dev)
        o = owlm(**inp)
        r = owlp.post_process_object_detection(o, threshold=0.05, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
        if len(r["scores"]) == 0:
            return None
        b = r["boxes"][int(r["scores"].argmax())].cpu().numpy()
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2     # cx, cy

    @torch.no_grad()
    def gdino(img, text):
        inp = gp(images=Image.fromarray(img), text=text, return_tensors="pt").to(dev)
        o = gm(**inp)
        r = gp.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=0.15,
                                                      text_threshold=0.15, target_sizes=[(H, W)])[0]
        if len(r["scores"]) == 0:
            return None
        b = r["boxes"][int(r["scores"].argmax())].cpu().numpy()
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def up(rc, dmap, c2w):
        return CU.transform_from_pixels_to_world(np.array(rc, float), dmap, c2w)[:3]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    res = {"owl": [[], [], 0], "gdino": [[], [], 0]}
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; tn = re.sub(r"_\d+$", "", T).replace("_", " ")
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T]); tp = body_pos(sim, rb[T]).astype(np.float64)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        proj = np.asarray(CU.project_points_from_world_to_camera(tp[None], w2c, H, W)[0])
        tr_f = H - 1 - int(np.clip(proj[0], 0, H - 1)); tc = int(np.clip(proj[1], 0, W - 1))
        line = f"  {pathlib.Path(bf).stem[:20]:22s} T={tn:13s}"
        for name, fn in (("owl", owl), ("gdino", gdino)):
            out = fn(img, f"a {tn}")
            if out is None:
                line += f" | {name}=MISS"; continue
            cx, cy = out; cx, cy = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
            px = float(np.hypot(cy - tr_f, cx - tc)); e3 = float(np.linalg.norm(up((cy, cx), rd, c2w) - tp))
            res[name][0].append(px); res[name][1].append(e3); res[name][2] += 1
            line += f" | {name} {px:.0f}px/{e3*100:.0f}cm"
        env.close()
        print(line, flush=True)
    print(f"\n=== STRONG DETECTOR COMPARISON (N={args.n}) ===", flush=True)
    for name in ("owl", "gdino"):
        px, e3, nd = res[name]
        m = lambda a: float(np.mean(a)) if a else float('nan')
        print(f"  {name:6s}: {m(px):.0f}px to true, {m(e3)*100:.0f}cm 3D, detected {nd}/{args.n}", flush=True)
    print("DET_COMPARE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
