"""Precision-binder test: OWLv2 box-center+patch-depth (current) vs OWLv2-box -> SAM mask ->
mask-centroid + median-depth-over-mask + geometry (proposed). Reports 3D binding error vs ground truth
for both, per named counterfactual target. Decides whether the mask binder fixes the bimodal 11-32cm errors.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/binder_sam_test.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    samp = SamProcessor.from_pretrained("facebook/sam-vit-base")
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
                b = r["boxes"].cpu().numpy()[sc.argmax()]
                best = (float(sc.max()), b)        # b = [x0,y0,x1,y1]
        return None if best is None else best[1]

    @torch.no_grad()
    def sam_mask(img, box):
        inp = samp(Image.fromarray(img), input_boxes=[[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]],
                   return_tensors="pt").to(dev)
        out = sam(**inp)
        masks = samp.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                        inp["reshaped_input_sizes"].cpu())[0][0]  # (3,H,W)
        iou = out.iou_scores.cpu().numpy().reshape(-1)
        return masks[int(iou.argmax())].numpy().astype(bool)

    def unproj(r, c, d, c2w):
        return np.asarray(CU.transform_from_pixels_to_world(np.array([r, c], float),
                                                            np.full((H, W, 1), d), c2w)[:3], np.float32)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    box_errs, sam_errs = [], []
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
        box = owl_box(img, nm(T))
        if box is None:
            env.close(); print(f"  {nm(T):15s} NO DETECTION"); continue
        # current: box center + 12px patch median depth
        cy, cx = int((box[1]+box[3])/2), int((box[0]+box[2])/2)
        d_box = float(np.median(rd[max(0,cy-6):cy+6, max(0,cx-6):cx+6, 0]))
        g_box = unproj(cy, cx, d_box, c2w)
        e_box = float(np.linalg.norm(g_box[:2] - obj[:2]))
        # proposed: SAM mask -> centroid + median depth over mask
        m = sam_mask(img, box)
        ys, xs = np.where(m)
        if len(ys) == 0:
            g_sam = g_box; e_sam = e_box
        else:
            my, mx = int(ys.mean()), int(xs.mean())
            d_mask = float(np.median(rd[ys, xs, 0]))
            g_sam = unproj(my, mx, d_mask, c2w)
            e_sam = float(np.linalg.norm(g_sam[:2] - obj[:2]))
        box_errs.append(e_box); sam_errs.append(e_sam)
        env.close()
        print(f"  {nm(T):15s} box={e_box*100:4.0f}cm  SAM-mask={e_sam*100:4.0f}cm  "
              f"{'BETTER' if e_sam < e_box - 0.02 else ('worse' if e_sam > e_box + 0.02 else 'same')}", flush=True)
    n = len(box_errs)
    print(f"\n=== BINDER: box vs SAM-mask (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  box-center+patch : mean {np.mean(box_errs)*100:.0f}cm  median {np.median(box_errs)*100:.0f}cm", flush=True)
    print(f"  SAM mask         : mean {np.mean(sam_errs)*100:.0f}cm  median {np.median(sam_errs)*100:.0f}cm", flush=True)
    print(f"  improved on      : {sum(1 for b,s in zip(box_errs,sam_errs) if s < b-0.02)}/{n}", flush=True)
    print("BINDER_SAM_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
