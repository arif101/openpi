"""Foveation perception ablation (research §3, no robot): does cropping+zooming let a recognizer disambiguate
the NAMED object among similar distractors? Render @1024, crop each object region at high res (object fills
the frame), and pick which crop matches the named phrase. Isolates resolution-vs-true-ambiguity.

Arms (top-1 disambiguation accuracy = chosen crop is the named target):
  CLIP-full   : full-frame CLIP sim of the WHOLE 1024 image to each name (sanity, ~chance)
  CLIP-crop   : per-object high-res crop -> CLIP sim to named phrase -> argmax  (does zoom+CLIP separate?)
  OWL-crop    : per-object crop -> OWLv2 named-query score -> argmax            (does zoom+detector separate?)
Uses GT object boxes to DEFINE crops (perception eval isolates recognition from proposal recall).
chance = 1/n_obj (~0.17); full-frame OWLv2 baseline (tonight) = ~0.40.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/foveation_test.py --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hi", type=int, default=1024); ap.add_argument("--win", type=float, default=0.13)
    ap.add_argument("--thr", type=float, default=0.0)
    args = ap.parse_args()
    import torch
    from transformers import (CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection)
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    H = args.hi; win = int(args.win * H)

    @torch.no_grad()
    def clip_sim(imgs, texts):
        ti = clipp(text=[f"a photo of {t}" for t in texts], images=imgs, return_tensors="pt",
                   padding=True).to(dev)
        o = clip(**ti)
        return o.logits_per_image.softmax(-1).cpu().numpy()    # [n_img, n_text]

    @torch.no_grad()
    def owl_score(crop, tgt):
        best = 0.0
        for q in (tgt, f"a {tgt}"):
            inp = owlp(text=[[q]], images=crop, return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[crop.size[1], crop.size[0]]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc): best = max(best, float(sc.max()))
        return best

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    acc = {"clip_full": [], "clip_crop": [], "owl_crop": []}
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        names = list(dict.fromkeys([T] + list(distractors)))
        tgt_idx = 0                                            # T is first
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=False)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rb = resolve_bodies(sim, names)
        crops = []
        for n in names:
            tp = CU.project_points_from_world_to_camera(body_pos(sim, rb[n])[None].astype(np.float32), w2c, H, H)[0]
            r, c = int(H - 1 - tp[0]), int(tp[1])
            r0, c0 = max(0, r-win), max(0, c-win); r1, c1 = min(H, r+win), min(H, c+win)
            crops.append(Image.fromarray(img[r0:r1, c0:c1]).resize((224, 224)))
        env.close()
        labels = [nm(n) for n in names]
        tphr = nm(T)
        # CLIP full-frame: whole image vs names
        sf = clip_sim([Image.fromarray(img).resize((224, 224))], labels)[0]
        acc["clip_full"].append(int(sf.argmax() == tgt_idx))
        # CLIP crop: each crop's sim to the NAMED phrase, pick best crop
        sc = clip_sim(crops, [tphr])[:, 0]
        acc["clip_crop"].append(int(sc.argmax() == tgt_idx))
        # OWL crop: each crop's OWLv2 score for the named query, pick best
        so = np.array([owl_score(cr, tphr) for cr in crops])
        acc["owl_crop"].append(int(so.argmax() == tgt_idx))
        print(f"  {tphr:15s} clip_full={'Y' if acc['clip_full'][-1] else '.'} "
              f"clip_crop={'Y' if acc['clip_crop'][-1] else '.'} owl_crop={'Y' if acc['owl_crop'][-1] else '.'} "
              f"(n_obj={len(names)})", flush=True)
    n = len(acc["clip_full"])
    print(f"\n=== FOVEATION disambiguation top-1 (N={n}, seed={args.seed}) ===", flush=True)
    for k in ("clip_full", "clip_crop", "owl_crop"):
        print(f"  {k:11s}: {np.mean(acc[k]):.2f}", flush=True)
    print(f"  chance ~0.17 ; full-frame OWLv2 (tonight) ~0.40", flush=True)
    print("FOVEATION_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
