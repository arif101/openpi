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


def masked_crop(up, m_up, cy, cx, R, half=60):
    """±half crop centered on the object, with NON-object pixels grayed (seg mask) -> appearance depends ONLY on
    the object, not background/neighbors (fixes look-alike flips from context leakage)."""
    y0, y1 = max(0, int(cy) - half), min(R, int(cy) + half); x0, x1 = max(0, int(cx) - half), min(R, int(cx) + half)
    crop = up[y0:y1, x0:x1].copy(); mc = m_up[y0:y1, x0:x1]
    crop[~mc] = 128
    return crop


def seg_proposals(seg_native, R, vflip=True, min_area=60):
    """Class-agnostic instance-seg -> EVERY object instance (idealized SAM: separates objects, no identity).
    Keep all non-background instances; DINOv2 rejects robot/non-objects at match time. Returns (cy_up, cx_up, mask_up)."""
    seg = np.asarray(seg_native).reshape(seg_native.shape[0], seg_native.shape[1])
    cents = []
    for i in np.unique(seg):
        if i == 0: continue                                         # background (table+floor)
        m = (seg == i); a = int(m.sum())
        if a < min_area: continue                                   # drop specks only
        ys, xs = np.where(m); cy, cx = ys.mean(), xs.mean()
        m_up = m[::-1] if vflip else m
        cy_up = (R - 1 - cy) if vflip else cy
        cents.append((cy_up, cx, m_up))
    return cents


def height_proposals(sim, depth_norm, R, cam, vflip=True, lift=0.015, maxh=0.5, min_area=200, max_area=40000):
    """GEOMETRY-based object proposals: unproject depth -> world height -> every blob ABOVE the table is an object.
    Perfect recall (anything sitting on the table), centroids ~ object centers. Returns (cy_up, cx_up, mask_up)."""
    import robosuite.utils.camera_utils as cu
    from scipy import ndimage
    real = cu.get_real_depth_map(sim, depth_norm)[..., 0]            # native HxW metric depth
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R); cam2world = np.linalg.inv(w2p)
    rows, cols = np.mgrid[0:R, 0:R]
    z = real
    cam_pts = np.stack([cols * z, rows * z, z, np.ones_like(z)], -1)  # HxWx4 (col=x,row=y per robosuite)
    world = cam_pts @ cam2world.T                                    # HxWx4 native-frame
    height = world[..., 2]
    table_z = np.percentile(height, 40)
    obj = (height > table_z + lift) & (height < table_z + maxh)      # objects above table, below arm/background
    lab, n = ndimage.label(obj)
    cents = []
    for i in range(1, n + 1):
        m = (lab == i); a = int(m.sum())
        if a < min_area or a > max_area: continue
        ys, xs = np.where(m); cy, cx = ys.mean(), xs.mean()          # native centroid
        m_up = m[::-1] if vflip else m
        cy_up = (R - 1 - cy) if vflip else cy
        cents.append((cy_up, cx, m_up))
    return cents


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


def build_bank_seg(bddls, init_dir, proto_inits, R, cam, dev, half=60):
    """Prototypes from the SAME seg-blob crops the query uses (crop-domain consistency). Each blob labeled offline
    by nearest projected true position (privileged labels for the catalog only; query stays non-privileged)."""
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    acc = {}
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        fi = pathlib.Path(init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        if not fi.exists(): continue
        inits = np.asarray(torch.load(fi, weights_only=False))
        for ti in proto_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True,
                                     camera_segmentations="instance")
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            up = np.asarray(obs[cam + "_image"])[::-1].copy()
            cents = seg_proposals(np.asarray(obs[cam + "_segmentation_instance"]), R, True)
            if not cents: env.close(); continue
            w2p = cu.get_camera_transform_matrix(sim, cam, R, R); rb = resolve_bodies(sim, objs)
            for o in objs:
                if rb[o] is None: continue
                px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
                tr_up = R - 1 - px[0]; tc = px[1]
                bi = int(np.argmin([(cy - tr_up) ** 2 + (cx - tc) ** 2 for (cy, cx, m) in cents]))
                cy, cx, m = cents[bi]
                if (cy - tr_up) ** 2 + (cx - tc) ** 2 > 60 ** 2: continue
                acc.setdefault(nm(o), []).append(dino_feat(masked_crop(up, m, cy, cx, R, half), dev))
            env.close()
    return {k: (np.mean(fs, 0) / np.linalg.norm(np.mean(fs, 0))) for k, fs in acc.items()}


def exemplar_bind(sim, rgb_native, depth_norm, target_name, all_names, bank, cam, R, dev, vflip=True, half=60, seg_native=None):
    import robosuite.utils.camera_utils as cu
    up = rgb_native[::-1] if vflip else rgb_native
    proto = bank.get(nm(target_name))
    if proto is None: return None, 0
    cents = seg_proposals(seg_native, R, vflip) if seg_native is not None else height_proposals(sim, depth_norm, R, cam, vflip)
    if not cents: return None, 0
    feats = [dino_feat(masked_crop(up, m, cy, cx, R, half), dev) for (cy, cx, m) in cents]  # object-only masked crops
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
    print(f"building masked seg prototype bank from inits {proto_inits}...", flush=True)
    bank = build_bank_seg(bddls, args.init_dir, proto_inits, R, args.cam, dev)
    print(f"bank has {len(bank)} objects: {list(bank)}", flush=True)
    berr = []; ndet = 0; ntot = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        for ti in query_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True,
                                     camera_segmentations="instance")
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            rb = resolve_bodies(sim, [T]); ntot += 1
            pred, nb = exemplar_bind(sim, np.asarray(obs[args.cam + "_image"]), np.asarray(obs[args.cam + "_depth"]),
                                     T, objs, bank, args.cam, R, dev,
                                     seg_native=np.asarray(obs[args.cam + "_segmentation_instance"]))
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
