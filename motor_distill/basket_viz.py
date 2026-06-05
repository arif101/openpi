"""Why does the foveation binder fail on the basket? Render@1024, run the SAME propose->crop->CLIP binder
for 'basket', and save annotated frame: all proposal boxes (orange), chosen basket pixel (red), true basket
(green). Shows whether it's a PROPOSAL miss (no box on basket) or a CLIP mispick (box exists, CLIP chose a grocery).
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hi", type=int, default=1024); ap.add_argument("--container", default="basket")
    args = ap.parse_args()
    import torch
    from transformers import CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image, ImageDraw
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    H = args.hi; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def propose(img, queries):
        boxes = []
        for q in queries:
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=0.0, target_sizes=torch.tensor([[H, H]]).to(dev))[0]
            for sc, b in zip(r["scores"].cpu().numpy(), r["boxes"].cpu().numpy()):
                boxes.append((float(sc), b))
        boxes.sort(key=lambda x: -x[0]); kept = []
        for sc, b in boxes:
            cy, cx = (b[1]+b[3])/2, (b[0]+b[2])/2
            if all(abs(cy-(k[1]+k[3])/2)+abs(cx-(k[0]+k[2])/2) > 0.13*H for k in kept): kept.append(b)
            if len(kept) >= 12: break
        return kept

    @torch.no_grad()
    def clip_scores(img, boxes, phrase):
        crops = []
        for b in boxes:
            x0,y0,x1,y1 = b; m = 0.15*max(x1-x0, y1-y0)+8
            r0,c0 = max(0,int(y0-m)), max(0,int(x0-m)); r1,c1 = min(H,int(y1+m)), min(H,int(x1+m))
            crops.append(Image.fromarray(img[r0:r1, c0:c1]).resize((224,224)))
        ti = clipp(text=[f"a photo of {phrase}"], images=crops, return_tensors="pt", padding=True).to(dev)
        return clip(**ti).logits_per_image.softmax(0)[:, 0].cpu().numpy()

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    for si, bf in enumerate(bddls):
        instr, objs, targets, distractors = parse_bddl(bf)
        names = list(dict.fromkeys([targets[0]] + list(distractors))) if targets else []
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=False)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        try:
            cb = resolve_bodies(sim, [args.container + "_1"])[args.container + "_1"]
            tp = CU.project_points_from_world_to_camera(body_pos(sim, cb)[None].astype(np.float32), w2c, H, H)[0]
            tr, tc = int(H-1-tp[0]), int(tp[1])
        except Exception:
            tr = tc = None
        qset = [nm(x) for x in names] + [args.container, "basket", "container", "bin"]
        boxes = propose(img, qset)
        sc = clip_scores(img, boxes, args.container)
        chosen = int(sc.argmax())
        im = Image.fromarray(img).resize((512, 512)); dr = ImageDraw.Draw(im); s = 512.0/H
        for i, b in enumerate(boxes):
            col = (255,0,0) if i == chosen else (255,160,0)
            dr.rectangle([b[0]*s, b[1]*s, b[2]*s, b[3]*s], outline=col, width=3 if i==chosen else 1)
            dr.text((b[0]*s, b[1]*s-10), f"{sc[i]:.2f}", fill=col)
        if tr is not None:
            dr.ellipse([tc*s-6, tr*s-6, tc*s+6, tr*s+6], fill=(0,255,0))
        im.save(f"logs/basketviz_{si}.png"); env.close()
        print(f"  scene {si}: {len(boxes)} proposals, CLIP-chose box {chosen} (score {sc[chosen]:.2f}); "
              f"saved logs/basketviz_{si}.png (red=chosen, green=true basket)", flush=True)
        if si >= 2: break
    print("BASKET_VIZ_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
