"""Resolution fix test: run the open-vocab binder at INCREASING render resolution. The objects are
distinguishable at high res (readable labels) but indistinguishable at 224 -- so feeding OWLv2 a high-res
image (we control LIBERO's camera resolution) should fix the named-target binding. Measures 3D bind error
+ <5cm count vs ground truth at res in {256, 512, 768, 1024}.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/binder_hires_test.py --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--res", type=int, nargs="+", default=[256, 512, 768])
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def locate(img, tgt, R):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[R, R]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]
                best = (float(sc.max()), int((b[1]+b[3])/2), int((b[0]+b[2])/2))
        return None if best is None else (best[1], best[2])

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    summary = {}
    for R in args.res:
        errs = []
        for bf in bddls:
            instr, objs, targets, distractors = parse_bddl(bf)
            if not targets: continue
            T = targets[0]
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
            env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
            rb = resolve_bodies(sim, [T]); obj = body_pos(sim, rb[T]).astype(np.float32)
            w2c = CU.get_camera_transform_matrix(sim, args.cam, R, R); c2w = np.linalg.inv(w2c)
            img = np.asarray(obs["agentview_image"])[::-1].copy()
            rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(R, R, 1))[::-1].copy()
            rc = locate(img, nm(T), R)
            if rc is None:
                errs.append(9.9); env.close(); continue
            cy, cx = rc; w = max(3, R // 42)
            d = float(np.median(rd[max(0,cy-w):cy+w, max(0,cx-w):cx+w, 0]))
            g = np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                             np.full((R, R, 1), d), c2w)[:3], np.float32)
            errs.append(float(np.linalg.norm(g[:2] - obj[:2])))
            env.close()
        errs = np.array(errs)
        summary[R] = errs
        print(f"  res {R:4d}: mean {errs[errs<9].mean()*100:4.0f}cm  median {np.median(errs)*100:4.0f}cm  "
              f"<5cm {int((errs<0.05).sum())}/{len(errs)}  miss {int((errs>9).sum())}", flush=True)
    print(f"\n=== BINDER vs RESOLUTION (N per res, seed={args.seed}) ===", flush=True)
    for R in args.res:
        e = summary[R]
        print(f"  {R:4d}px: <5cm {int((e<0.05).sum())}/{len(e)}  median {np.median(e)*100:.0f}cm", flush=True)
    print("BINDER_HIRES_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
