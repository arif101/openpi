"""FOVEATION BINDER (training-free, no privileged identity): the naive single-box detector hits the resolution
wall (returns the salient basket for every query). Fix = propose ALL object regions -> crop each at hi-res ->
CLIP-disambiguate against the scene's object names -> unproject the winning box -> 3D goal.

Geometry already validated (round-trip 3.6cm). This module adds the WHICH-object perception.
--validate: measure predicted-vs-TRUE 3D error + which-object accuracy (no motor). Run FIRST.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/bind_foveate.py \
       --bddl-dir /root/LIBERO-PRO/libero/libero/bddl_files/libero_object \
       --init-dir /root/LIBERO-PRO/libero/libero/init_files/libero_object --res 512 --n 10 --trials 2 --validate
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos

_M = {}


def nm(o):  # bddl body name -> readable phrase
    import re
    return re.sub(r"_\d+$", "", o).replace("_", " ")


def detect_all(img, query, device, box_thr=0.12, text_thr=0.12):
    """ALL boxes above threshold for `query` (phrases separated by '. '). Returns list of [x0,y0,x1,y1]."""
    from PIL import Image
    if "gd" not in _M:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        mid = "IDEA-Research/grounding-dino-base"
        _M["gp"] = AutoProcessor.from_pretrained(mid)
        _M["gd"] = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    proc, model = _M["gp"], _M["gd"]
    pil = Image.fromarray(img)
    inp = proc(images=pil, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model(**inp)
    res = proc.post_process_grounded_object_detection(o, inp.input_ids, box_threshold=box_thr,
                                                      text_threshold=text_thr, target_sizes=[pil.size[::-1]])[0]
    return [b.cpu().numpy().astype(int) for b in res["boxes"]]


def clip_scores(crops, texts, device):
    """cosine sim [n_crops, n_texts] via CLIP."""
    from PIL import Image
    if "clip" not in _M:
        from transformers import CLIPModel, CLIPProcessor
        mid = "openai/clip-vit-large-patch14"
        _M["clip"] = CLIPModel.from_pretrained(mid).to(device).eval()
        _M["clipp"] = CLIPProcessor.from_pretrained(mid)
    model, proc = _M["clip"], _M["clipp"]
    pcs = [Image.fromarray(c) for c in crops]
    inp = proc(text=[f"a photo of {t}" for t in texts], images=pcs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits_per_image.softmax(-1).cpu().numpy()   # [n_crops, n_texts]


def foveate_bind(sim, rgb_native, depth_norm, target, all_names, cam, H, W, device, vflip=True, pad=4):
    """Return (3D world point, chosen_label) for `target` among `all_names`. Training-free."""
    import robosuite.utils.camera_utils as cu
    up = rgb_native[::-1] if vflip else rgb_native        # upright for the detector
    phrases = [nm(o) for o in all_names]
    boxes = detect_all(up, " . ".join(phrases) + " .", device)
    if not boxes:
        boxes = detect_all(up, "an object . a bottle . a box . a can . a carton .", device)
    if not boxes:
        return None, None
    crops, keep = [], []
    for b in boxes:
        x0, y0, x1, y1 = b
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
        if x1 - x0 < 4 or y1 - y0 < 4: continue
        crops.append(up[y0:y1, x0:x1]); keep.append(b)
    if not crops:
        return None, None
    S = clip_scores(crops, phrases, device)                # [nbox, nname]
    ti = all_names.index(target)
    assigned = S.argmax(1)
    cand = [i for i in range(len(keep)) if assigned[i] == ti]
    bi = (max(cand, key=lambda i: S[i, ti]) if cand else int(S[:, ti].argmax()))
    x0, y0, x1, y1 = keep[bi]
    cu_y = (y0 + y1) / 2; cu_x = (x0 + x1) / 2             # row,col in UPRIGHT frame
    r_native = (H - 1 - cu_y) if vflip else cu_y           # map back to native for depth/geometry
    real = cu.get_real_depth_map(sim, depth_norm)
    w2p = cu.get_camera_transform_matrix(sim, cam, H, W); cam2world = np.linalg.inv(w2p)
    pts = cu.transform_from_pixels_to_world(np.array([[r_native, cu_x]]), real[None], cam2world)
    return np.asarray(pts[0], np.float32), phrases[bi if not cand else bi]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", default="")
    p.add_argument("--cam", default="agentview"); p.add_argument("--res", type=int, default=512)
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=2)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--seed", type=int, default=20)
    p.add_argument("--vflip", type=int, default=1); p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    berr = []; nright = 0; ndet = 0; ntot = 0; R = args.res
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        all_names = objs                                   # every scene object (incl distractors + container)
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, (len(inits) - s0)) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
            env.seed(args.seed + ti); env.reset(); obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            sim = env.env.sim; rb = resolve_bodies(sim, [T])
            rgb = np.asarray(obs[args.cam + "_image"]); dep = np.asarray(obs[args.cam + "_depth"])
            pred, lab = foveate_bind(sim, rgb, dep, T, all_names, args.cam, R, R, dev, vflip=bool(args.vflip))
            ntot += 1
            if pred is not None:
                ndet += 1; true = body_pos(sim, rb[T]); e = float(np.linalg.norm(pred - true)); berr.append(e)
                ok = (e < 0.06); nright += int(ok)
                print(f"  {stem[:28]:30s} -> '{lab}' err={e*100:.1f}cm {'OK' if ok else 'x'}", flush=True)
            else:
                print(f"  {stem[:28]:30s} NO BIND", flush=True)
            env.close()
    med = np.median(berr) if berr else float("nan")
    print(f"\n=== FOVEATION BINDER (res={R}): detect {ndet}/{ntot}, within-6cm {nright}/{ntot}, "
          f"median 3D err {med*100:.1f}cm (mean {np.mean(berr)*100 if berr else 0:.1f}cm) ===", flush=True)
    print("FOVEATE_VALIDATE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
