"""Save annotated frames to SEE what OWLv2 binds for each named counterfactual target:
red box = OWLv2 detection, green dot = TRUE object projected pixel, label = 3D bind error.
Tells us if the bad-half binding errors are wrong-object (detection) or wrong-geometry (box on object
but unprojects off). -> saves /root/openpi/logs/bind_<target>.png
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image, ImageDraw
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def owl_all(img, tgt):
        boxes = []
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            for sc, b in zip(r["scores"].cpu().numpy(), r["boxes"].cpu().numpy()):
                boxes.append((float(sc), b))
        return boxes

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); obj = body_pos(sim, rb[T]).astype(np.float32)
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        tp = CU.project_points_from_world_to_camera(obj[None], w2c, H, W)[0]
        tr, tc = int(H - 1 - tp[0]), int(tp[1])
        boxes = owl_all(img, nm(T))
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        boxes.sort(key=lambda x: -x[0])
        for i, (sc, b) in enumerate(boxes[:4]):
            col = (255, 0, 0) if i == 0 else (255, 160, 0)
            dr.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline=col, width=2)
            dr.text((float(b[0]), float(b[1]) - 8), f"{sc:.2f}", fill=col)
        dr.ellipse([tc - 4, tr - 4, tc + 4, tr + 4], fill=(0, 255, 0))   # true object pixel (green)
        safe = nm(T).replace(" ", "_")
        im.save(f"{args.out}/bind_{safe}.png")
        env.close()
        print(f"  saved bind_{safe}.png  (target='{nm(T)}', {len(boxes)} dets, green=true obj)", flush=True)
    print("BINDER_VIZ_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
