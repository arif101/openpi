"""Better-grounding test: GroundingDINO vs OWLv2 for the named counterfactual target. OWLv2 confidently
mis-grounds ~half of LIBERO's small grocery objects (boxes a distractor/basket). Test whether
GroundingDINO grounds them more accurately. Reports 3D bind error vs ground truth for both, same frames.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/binder_gdino_test.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--gd", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()
    import torch
    from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                              AutoProcessor, GroundingDinoForObjectDetection)
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    gdp = AutoProcessor.from_pretrained(args.gd)
    gdm = GroundingDinoForObjectDetection.from_pretrained(args.gd).to(dev).eval()
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def owl_box(img, tgt):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                best = (float(sc.max()), r["boxes"].cpu().numpy()[sc.argmax()])
        return None if best is None else best[1]

    @torch.no_grad()
    def gd_box(img, tgt):
        inp = gdp(images=Image.fromarray(img), text=f"{tgt}.", return_tensors="pt").to(dev)
        out = gdm(**inp)
        r = gdp.post_process_grounded_object_detection(out, inp.input_ids, box_threshold=0.2,
                                                       text_threshold=0.2, target_sizes=[(H, W)])[0]
        sc = r["scores"].cpu().numpy()
        if not len(sc): return None
        return r["boxes"].cpu().numpy()[sc.argmax()]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    oe, ge = [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); obj = body_pos(sim, rb[T]).astype(np.float32)
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()

        def err_of(b):
            if b is None: return 9.9
            cy, cx = int((b[1]+b[3])/2), int((b[0]+b[2])/2)
            d = float(np.median(rd[max(0,cy-6):cy+6, max(0,cx-6):cx+6, 0]))
            g = np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                             np.full((H, W, 1), d), c2w)[:3], np.float32)
            return float(np.linalg.norm(g[:2] - obj[:2]))

        e_o = err_of(owl_box(img, nm(T))); e_g = err_of(gd_box(img, nm(T)))
        oe.append(e_o); ge.append(e_g)
        env.close()
        tag = 'GDINO-better' if e_g < e_o - 0.02 else ('OWL-better' if e_o < e_g - 0.02 else 'same')
        print(f"  {nm(T):15s} OWLv2={e_o*100:4.0f}cm  GDINO={e_g*100:4.0f}cm  {tag}", flush=True)
    n = len(oe)
    print(f"\n=== BINDER: OWLv2 vs GroundingDINO ({args.gd}, N={n}, seed={args.seed}) ===", flush=True)
    print(f"  OWLv2 : mean {np.mean(oe)*100:.0f}cm median {np.median(oe)*100:.0f}cm  <5cm {sum(e<0.05 for e in oe)}/{n}", flush=True)
    print(f"  GDINO : mean {np.mean(ge)*100:.0f}cm median {np.median(ge)*100:.0f}cm  <5cm {sum(e<0.05 for e in ge)}/{n}", flush=True)
    print("BINDER_GDINO_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
