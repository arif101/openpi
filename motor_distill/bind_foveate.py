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


def sam_clean_crop(up, box, device, pad=4):
    """SAM mask within the proposal box -> single-object crop with background grayed (research lever #1:
    clean crops are what make the fine-grained encoder work; loose boxes defeat it)."""
    from PIL import Image
    if "sam" not in _M:
        from transformers import SamModel, SamProcessor
        _M["sam"] = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
        _M["samp"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam, sp = _M["sam"], _M["samp"]
    H, W = up.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pil = Image.fromarray(up)
    inp = sp(pil, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = sam(**inp)
    masks = sp.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                  inp["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores.cpu().numpy()[0, 0]
    m = masks[int(scores.argmax())].numpy().astype(bool)               # best mask, HxW
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad); cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = up[cy0:cy1, cx0:cx1].copy(); mc = m[cy0:cy1, cx0:cx1]
    crop[~mc] = 128                                                    # gray background
    return crop, m


def clip_scores(crops, texts, device):
    """[n_crops, n_texts] crop-vs-name scores via FG-CLIP (fine-grained-specialized encoder, trained on 10M hard
    fine-grained negatives). Deep-research verdict: the recognition step is the proven bottleneck (oracle masks +
    CLIP=20% vs oracle classifier=66%); FG-CLIP targets exactly the similar-object disambiguation failure."""
    from PIL import Image
    if "fg" not in _M:
        from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
        mid = "qihoo360/fg-clip-base"
        _M["fg"] = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).to(device).eval()
        _M["fgt"] = AutoTokenizer.from_pretrained(mid)
        _M["fgi"] = AutoImageProcessor.from_pretrained(mid)
    m, tok, ip = _M["fg"], _M["fgt"], _M["fgi"]
    pcs = [Image.fromarray(c) for c in crops]
    pix = ip(images=pcs, return_tensors="pt").pixel_values.to(device)
    ids = torch.tensor(tok([f"a photo of {t}" for t in texts], max_length=77, padding="max_length",
                           truncation=True).input_ids).to(device)
    with torch.no_grad():
        imf = m.get_image_features(pix); txf = m.get_text_features(ids, walk_short_pos=True)
    imf = imf / imf.norm(dim=-1, keepdim=True); txf = txf / txf.norm(dim=-1, keepdim=True)
    return (imf @ txf.T).cpu().numpy()                     # [n_crops, n_texts]


def foveate_bind(sim, rgb_native, depth_norm, target, all_names, cam, H, W, device, vflip=True, pad=4, clean=True):
    """Return (3D world point, chosen_label, nbox) for `target` among `all_names`. Training-free.
    clean=True: SAM-mask each proposal -> single-object crop (research lever #1) + mask-based 3D center."""
    import robosuite.utils.camera_utils as cu
    up = rgb_native[::-1] if vflip else rgb_native        # upright for the detector
    phrases = [nm(o) for o in all_names]
    boxes = detect_all(up, " . ".join(phrases) + " .", device)
    if not boxes:
        boxes = detect_all(up, "an object . a bottle . a box . a can . a carton .", device)
    if not boxes:
        return None, None, 0
    crops, keep, masks = [], [], []
    for b in boxes:
        x0, y0, x1, y1 = b
        if x1 - x0 < 4 or y1 - y0 < 4: continue
        if clean:
            crop, m = sam_clean_crop(up, b, device, pad)              # SAM single-object crop (gray bg)
        else:
            xa, ya = max(0, x0 - pad), max(0, y0 - pad); xb, yb = min(W, x1 + pad), min(H, y1 + pad)
            crop, m = up[ya:yb, xa:xb], None
        crops.append(crop); keep.append(b); masks.append(m)
    if not crops:
        return None, None, 0
    S = clip_scores(crops, phrases, device)                # [nbox, nname]
    ti = all_names.index(target)
    assigned = S.argmax(1)
    def area(i):
        x0, y0, x1, y1 = keep[i]; return float((x1 - x0) * (y1 - y0))
    cand = [i for i in range(len(keep)) if assigned[i] == ti]
    # localization: among boxes CLIP labels as the target, prefer a TIGHT box (loose/oversized boxes
    # cover several objects -> bad center). Tie-break by CLIP target-confidence.
    if cand:
        amax = max(area(i) for i in cand)
        tight = [i for i in cand if area(i) <= 0.5 * amax] or cand
        bi = max(tight, key=lambda i: S[i, ti])
    else:
        bi = int(S[:, ti].argmax())
    x0, y0, x1, y1 = keep[bi]                              # box in UPRIGHT frame
    real = cu.get_real_depth_map(sim, depth_norm)          # native HxWx1
    w2p = cu.get_camera_transform_matrix(sim, cam, H, W); cam2world = np.linalg.inv(w2p)
    m = masks[bi]
    if m is not None and m.any():
        m_nat = m[::-1] if vflip else m                    # SAM mask -> native frame
        ys, xs = np.where(m_nat)
        rr, cc = ys.mean(), xs.mean()                      # object-mask centroid = object center pixel
    else:
        # fallback: box near-surface (closest pixels = object, not table behind a thin bottle)
        ry0, ry1 = (H - 1 - y1, H - 1 - y0) if vflip else (y0, y1)
        ry0, ry1 = max(0, ry0), min(H, ry1); cx0, cx1 = max(0, x0), min(W, x1)
        patch = real[ry0:ry1, cx0:cx1, 0]
        if patch.size:
            om = patch <= np.percentile(patch, 25) + 0.03
            ys, xs = np.where(om)
            rr, cc = (ry0 + ys.mean(), cx0 + xs.mean()) if len(ys) else ((ry0 + ry1) / 2, (cx0 + cx1) / 2)
        else:
            rr, cc = (ry0 + ry1) / 2, (cx0 + cx1) / 2
    pts = cu.transform_from_pixels_to_world(np.array([[rr, cc]]), real[None], cam2world)
    chosen = phrases[int(S[bi].argmax())]                  # what the disambiguator thinks the box is
    return np.asarray(pts[0], np.float32), chosen, len(keep)


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
            pred, lab, nbox = foveate_bind(sim, rgb, dep, T, all_names, args.cam, R, R, dev, vflip=bool(args.vflip))
            ntot += 1
            if pred is not None:
                ndet += 1; true = body_pos(sim, rb[T]); e = float(np.linalg.norm(pred - true)); berr.append(e)
                ok = (e < 0.06); nright += int(ok)
                print(f"  {stem[:24]:26s} want='{nm(T)}' got='{lab}' nbox={nbox} err={e*100:.1f}cm {'OK' if ok else 'x'}", flush=True)
            else:
                print(f"  {stem[:24]:26s} NO BIND (nbox={nbox})", flush=True)
            env.close()
    med = np.median(berr) if berr else float("nan")
    print(f"\n=== FOVEATION BINDER (res={R}): detect {ndet}/{ntot}, within-6cm {nright}/{ntot}, "
          f"median 3D err {med*100:.1f}cm (mean {np.mean(berr)*100 if berr else 0:.1f}cm) ===", flush=True)
    print("FOVEATE_VALIDATE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
