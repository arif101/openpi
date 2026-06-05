"""Closed-set grounding test. The binding failure is OWLv2 mis-grounding the NAMED target among similar
distractors (it boxes a distractor/adjacent object). Fix: query OWLv2 with ALL scene object names and assign
detections competitively (greedy 1-to-1 by score), so each distractor claims its own box and the target
can't steal one. Uses only the scene object VOCABULARY (legitimately available), not positions.

Compares single-query (current) vs closed-set 3D bind error vs ground truth.
Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/binder_closedset_test.py --n 12 --seed 7
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
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def best_box(img, tgt):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]
                best = (float(sc.max()), b)
        return best   # (score, box[x0,y0,x1,y1]) or None

    def ctr(b):
        return int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)

    def iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, x1 - x0), max(0, y1 - y0); inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    single_errs, closed_errs = [], []
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
            cy, cx = ctr(b); d = float(np.median(rd[max(0,cy-6):cy+6, max(0,cx-6):cx+6, 0]))
            g = np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                             np.full((H, W, 1), d), c2w)[:3], np.float32)
            return float(np.linalg.norm(g[:2] - obj[:2]))

        # single-query (current)
        sb = best_box(img, nm(T))
        e_single = err_of(sb[1]) if sb else 9.9
        # closed-set: detect all names, greedy 1-to-1 assignment by score over overlapping boxes
        cand = [nm(o) for o in ([T] + list(distractors))]
        dets = []
        for name in cand:
            bb = best_box(img, name)
            if bb: dets.append((bb[0], name, bb[1]))
        dets.sort(key=lambda x: -x[0])
        assigned = {}; taken = []
        for sc, name, b in dets:
            if name in assigned: continue
            if any(iou(b, tb) > 0.5 for tb in taken):     # this region already claimed by a higher-score name
                continue
            assigned[name] = b; taken.append(b)
        tb = assigned.get(nm(T))
        e_closed = err_of(tb) if tb is not None else e_single
        single_errs.append(e_single); closed_errs.append(e_closed)
        env.close()
        tag = 'BETTER' if e_closed < e_single - 0.02 else ('worse' if e_closed > e_single + 0.02 else 'same')
        print(f"  {nm(T):15s} single={e_single*100:4.0f}cm  closed-set={e_closed*100:4.0f}cm  {tag}", flush=True)
    n = len(single_errs)
    print(f"\n=== BINDER: single-query vs closed-set (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  single-query : mean {np.mean(single_errs)*100:.0f}cm  median {np.median(single_errs)*100:.0f}cm", flush=True)
    print(f"  closed-set   : mean {np.mean(closed_errs)*100:.0f}cm  median {np.median(closed_errs)*100:.0f}cm", flush=True)
    print(f"  <5cm count   : single {sum(e<0.05 for e in single_errs)}/{n}  closed {sum(e<0.05 for e in closed_errs)}/{n}", flush=True)
    print("BINDER_CLOSEDSET_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
