"""Option B: open-vocab detector (Grounding-DINO) -> object pixel -> fitted
homography -> world (x,y). The decisive test of whether a PURPOSE-BUILT localizer
breaks the ~3.6cm frozen-Pi0.5-feature wall.

Seven readouts on Pi0.5's frozen features all plateaued at ~3.6cm held-out (it's
the features, not the readout). A detector trained to localize should hit the
object center to a few pixels = sub-cm on a tabletop, AND generalize to novel
objects. We localize "bowl" in each corpus base image, fit a pixel->world(x,y)
homography on TRAIN traces, and measure held-out world error vs the 3.6cm wall.

Camera convention is sidestepped: the homography is fit from real (detector-pixel,
known-world) pairs, so it learns the exact image->table-plane map from data.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import numpy as np


N_FRAMES = 6
QUERY = "a bowl."


def detect_pixels(files, query, device):
    """Return per-frame (pixel[u,v], world_xyz, trace_id, detected?)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    import rekey

    px, wl, tid, ok = [], [], [], []
    for ti, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        imgs, it, op = d["image"], d["image_t"], d["object_pos"]
        names = list(d["object_names"]); tobj = rekey.select_target_object(op, names)
        if imgs.shape[0] == 0:
            continue
        k = min(N_FRAMES, imgs.shape[0])
        for j in range(k):
            img = imgs[j]                                          # 256x256 natural orientation
            t0 = int(np.clip(it[j], 0, op.shape[0] - 1))
            pil = Image.fromarray(img)
            inp = proc(images=pil, text=query, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            res = proc.post_process_grounded_object_detection(
                out, inp.input_ids, box_threshold=0.20, text_threshold=0.20,
                target_sizes=[pil.size[::-1]])[0]
            wl.append(op[t0, tobj].astype(np.float64)); tid.append(ti)
            if len(res["scores"]) == 0:
                px.append([np.nan, np.nan]); ok.append(False)
            else:
                b = res["boxes"][int(res["scores"].argmax())].cpu().numpy()  # xyxy
                px.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); ok.append(True)
        if ti % 20 == 0:
            print(f"  detect trace {ti}/{len(files)}", flush=True)
    return np.array(px), np.array(wl), np.array(tid), np.array(ok)


def fit_homography(px, xy):
    """DLT: pixel (u,v) -> world (x,y) projective. px[N,2], xy[N,2] -> H 3x3."""
    A, b = [], []
    for (u, v), (x, y) in zip(px, xy):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); b.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); b.append(y)
    h = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def apply_homography(H, px):
    uv1 = np.concatenate([px, np.ones((px.shape[0], 1))], 1)
    p = uv1 @ H.T
    return p[:, :2] / p[:, 2:3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/keystone")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--query", default=QUERY)
    args = p.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = []
    for pert in (0, 5, 10):
        files += sorted(glob.glob(str(pathlib.Path(args.data) / f"pert{pert}" /
                        f"PHYS_OK_libero_10_task3_*baseline*.npz")))
    print(f"{len(files)} traces, query='{args.query}', device={dev}", flush=True)
    px, wl, tid, ok = detect_pixels(files, args.query, dev)
    print(f"detection rate: {ok.mean()*100:.0f}% ({ok.sum()}/{len(ok)})", flush=True)

    # split by trace
    rng = np.random.default_rng(args.seed)
    utid = np.unique(tid); rng.shuffle(utid)
    n_te = max(1, int(len(utid) * 0.2)); te_ids = set(utid[:n_te].tolist())
    te = np.array([t in te_ids for t in tid]); tr = ~te
    trm = tr & ok; tem = te & ok                                  # use detected frames only
    H = fit_homography(px[trm], wl[trm, :2])
    pred_xy = apply_homography(H, px[tem])
    err_xy = np.linalg.norm(pred_xy - wl[tem, :2], axis=-1)
    zc = wl[trm, 2].mean()                                        # table-plane z prior
    err_3d = np.sqrt(err_xy ** 2 + (wl[tem, 2] - zc) ** 2)
    base = np.linalg.norm(wl[tem, :2] - wl[trm, :2].mean(0), axis=-1)

    print(f"\n=== OPTION B: Grounding-DINO + fitted homography (held-out, detected frames) ===", flush=True)
    print(f"  detection rate      : {ok.mean()*100:.0f}%", flush=True)
    print(f"  world (x,y) error   : mean {err_xy.mean()*100:.1f}cm  median {np.median(err_xy)*100:.1f}cm", flush=True)
    print(f"  world 3D error      : mean {err_3d.mean()*100:.1f}cm  median {np.median(err_3d)*100:.1f}cm", flush=True)
    print(f"  predict-mean (x,y)  : mean {base.mean()*100:.1f}cm", flush=True)
    print(f"  vs frozen-feature wall: 3.6cm 3D mean / 1.7cm median", flush=True)
    win = err_3d.mean() < 0.025
    print(f"  -> {'B BREAKS THE WALL: detector localizes -> world encoder viable' if win else 'no decisive win'}", flush=True)
    print("OPTIONB_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
