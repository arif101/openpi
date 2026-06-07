"""EXEMPLAR-CORRESPONDENCE BINDER (general, text-free). Diagnostic showed DINOv2 separates the LIBERO objects
1.00 -> the ~50% wall was the CLIP/SigLIP/VLM TEXT-alignment, not visual ambiguity. So: build a per-object visual
PROTOTYPE bank (an object catalog / support set -- in deployment from product images; here from a few REFERENCE
inits, disjoint from test), then at test time match NON-privileged proposal crops to the target's prototype by
DINOv2 cosine. General/open-vocab (any object with a reference; no position memorization; frozen model, no training).

--validate: build bank from --proto-inits, test 3D bind on --query-inits (disjoint), proposals NON-privileged.
Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/bind_exemplar.py \
       --res 1024 --proto-inits 0,2,4 --query-inits 20,22 --validate
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from bind_foveate import detect_all, sam_clean_crop, nm
from dino_separability import dino_feat

_MX = {}


def sam_everything_centers(up, device, min_area=1500, max_area=22000, min_compact=0.35, dedup_px=35, pps=24):
    """Class-agnostic SAM 'segment everything' -> OBJECT-LEVEL centroids+masks (perfect recall, no detector text).
    Filter to object-sized COMPACT blobs (not table/parts/whole-scene) + dedup nested masks (keep object granularity)."""
    from PIL import Image
    if "gen" not in _MX:
        from transformers import pipeline
        _MX["gen"] = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if device == "cuda" else -1)
    out = _MX["gen"](Image.fromarray(up), points_per_side=pps, points_per_batch=64)
    cand = []
    for m in out["masks"]:
        m = np.asarray(m); a = int(m.sum())
        if a < min_area or a > max_area: continue
        ys, xs = np.where(m); h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
        if a / float(h * w) < min_compact: continue                 # compact blob, not sparse/elongated table strip
        cand.append((a, float(ys.mean()), float(xs.mean()), m))
    cand.sort(reverse=True, key=lambda c: c[0])                     # largest first
    kept = []
    for a, cy, cx, m in cand:                                       # dedup nested/duplicate masks by centroid
        if all((cy - k[0]) ** 2 + (cx - k[1]) ** 2 > dedup_px ** 2 for k in kept):
            kept.append((cy, cx, m))
    return kept


def proto_crop(sim, rgb_up, R, cam, body, half=60):
    import robosuite.utils.camera_utils as cu
    pos = body_pos(sim, body); w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]
    r_up = int(R - 1 - px[0]); c = int(px[1])
    y0, y1 = max(0, r_up - half), min(R, r_up + half); x0, x1 = max(0, c - half), min(R, c + half)
    return rgb_up[y0:y1, x0:x1] if (y1 - y0 > 8 and x1 - x0 > 8) else None


def build_bank(bddls, init_dir, proto_inits, R, cam, dev, half=60):
    """Per-object DINOv2 prototype from reference inits (offline catalog)."""
    bank = {}
    from libero.libero.envs import OffScreenRenderEnv
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = np.asarray(torch.load(pathlib.Path(init_dir) / f"{stem}.pruned_init", weights_only=False))
        fs = []
        for ti in proto_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            up = np.asarray(obs[cam + "_image"])[::-1].copy(); rb = resolve_bodies(sim, [T])
            c = proto_crop(sim, up, R, cam, rb[T], half)
            if c is not None: fs.append(dino_feat(c, dev))
            env.close()
        if fs:
            v = np.mean(fs, 0); bank[nm(T)] = v / np.linalg.norm(v)
    return bank


def exemplar_bind(sim, rgb_native, depth_norm, target_name, all_names, bank, cam, R, dev, vflip=True, half=60):
    import robosuite.utils.camera_utils as cu
    up = rgb_native[::-1] if vflip else rgb_native
    proto = bank.get(nm(target_name))
    if proto is None: return None, 0
    cents = sam_everything_centers(up, dev)                        # class-agnostic perfect-recall proposals
    if not cents: return None, 0
    feats = []
    for (cy, cx, m) in cents:
        y0, y1 = max(0, int(cy) - half), min(R, int(cy) + half); x0, x1 = max(0, int(cx) - half), min(R, int(cx) + half)
        feats.append(dino_feat(up[y0:y1, x0:x1], dev))             # fixed-scale crop matching prototype style
    bi = int((np.stack(feats) @ proto).argmax())                  # DINOv2 visual match
    cy, cx, m = cents[bi]                                          # winning object mask
    m_nat = m[::-1] if vflip else m; ys, xs = np.where(m_nat); rr, cc = ys.mean(), xs.mean()
    real = cu.get_real_depth_map(sim, depth_norm)
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); cam2world = np.linalg.inv(w2p)
    pts = cu.transform_from_pixels_to_world(np.array([[rr, cc]]), real[None], cam2world)
    return np.asarray(pts[0], np.float32), len(cents)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--cam", default="agentview"); p.add_argument("--res", type=int, default=1024)
    p.add_argument("--proto-inits", default="0,2,4"); p.add_argument("--query-inits", default="20,22")
    p.add_argument("--n", type=int, default=10); p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    query_inits = [int(x) for x in args.query_inits.split(",")]
    print(f"building prototype bank from inits {proto_inits}...", flush=True)
    bank = build_bank(bddls, args.init_dir, proto_inits, R, args.cam, dev)
    print(f"bank has {len(bank)} objects: {list(bank)}", flush=True)
    berr = []; ndet = 0; ntot = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        for ti in query_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            rb = resolve_bodies(sim, [T]); ntot += 1
            pred, nb = exemplar_bind(sim, np.asarray(obs[args.cam + "_image"]), np.asarray(obs[args.cam + "_depth"]),
                                     T, objs, bank, args.cam, R, dev)
            if pred is not None:
                true = body_pos(sim, rb[T]); e = float(np.linalg.norm(pred - true)); berr.append(e); ndet += 1
                print(f"  {stem[:24]:26s} i{ti} err={e*100:.1f}cm {'OK' if e<0.06 else 'x'} nbox={nb}", flush=True)
            else:
                print(f"  {stem[:24]:26s} i{ti} NO BIND", flush=True)
            env.close()
    med = np.median(berr) if berr else float("nan")
    print(f"\n=== EXEMPLAR (DINOv2) BINDER: det {ndet}/{ntot}, within-6cm {sum(e<0.06 for e in berr)}/{ntot}, "
          f"median {med*100:.1f}cm ===  [crop-CLIP/VLM wall: ~50%, 13cm]", flush=True)
    print("EXEMPLAR_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
