"""DECISIVE diagnostic: are the LIBERO objects visually SEPARABLE by self-supervised visual features (DINOv2),
independent of text? If yes -> an exemplar/correspondence binder (match scene crop to a visual reference, NOT to
CLIP text) can work. If even clean privileged crops aren't separable -> objects are genuinely ambiguous at this
resolution -> pivot to active perception. Uses PRIVILEGED positions to crop each object (this is a separability
test, not the binder). Cross-init nearest-prototype object-ID accuracy + which objects confuse.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/dino_separability.py \
       --res 1024 --proto-inits 20,22,24 --query-inits 26,28,30 --half 60
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos

_M = {}


def dino_feat(crop, device, mid="facebook/dinov2-large"):
    from PIL import Image
    if "d" not in _M:
        from transformers import AutoImageProcessor, AutoModel
        _M["d"] = AutoModel.from_pretrained(mid).to(device).eval()
        _M["dp"] = AutoImageProcessor.from_pretrained(mid)
    model, proc = _M["d"], _M["dp"]
    inp = proc(images=Image.fromarray(crop), return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    f = out.pooler_output[0]                          # CLS pooled
    return (f / f.norm()).cpu().numpy()


def crop_at(sim, rgb_up, R, cam, body, half):
    import robosuite.utils.camera_utils as cu
    pos = body_pos(sim, body)
    w2p = cu.get_camera_transform_matrix(sim, cam, R, R)
    px = cu.project_points_from_world_to_camera(pos[None], w2p, R, R)[0]   # native row,col
    r_up = int(R - 1 - px[0]); c = int(px[1])
    y0, y1 = max(0, r_up - half), min(R, r_up + half); x0, x1 = max(0, c - half), min(R, c + half)
    if y1 - y0 < 8 or x1 - x0 < 8: return None
    return rgb_up[y0:y1, x0:x1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--res", type=int, default=1024); p.add_argument("--cam", default="agentview")
    p.add_argument("--proto-inits", default="20,22,24"); p.add_argument("--query-inits", default="26,28,30")
    p.add_argument("--half", type=int, default=60)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"; R = args.res
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    query_inits = [int(x) for x in args.query_inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    # gather each TARGET object's crops across inits (privileged crop)
    feats = {}  # name -> {init: feat}
    names = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; stem = pathlib.Path(bf).stem; names.append(T)
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        feats[T] = {}
        for ti in proto_inits + query_inits:
            if ti >= len(inits): continue
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
            env.seed(ti); env.reset(); obs = env.set_init_state(inits[ti]); sim = env.env.sim
            rgb_up = np.asarray(obs[args.cam + "_image"])[::-1].copy()
            rb = resolve_bodies(sim, [T]); crop = crop_at(sim, rgb_up, R, args.cam, rb[T], args.half)
            if crop is not None: feats[T][ti] = dino_feat(crop, dev)
            env.close()
        print(f"  encoded {T}", flush=True)
    # prototypes = mean over proto_inits; classify query crops by nearest prototype
    protos = {n: np.mean([feats[n][ti] for ti in proto_inits if ti in feats[n]], 0) for n in names if any(ti in feats[n] for ti in proto_inits)}
    protos = {n: v / np.linalg.norm(v) for n, v in protos.items()}
    P = np.stack([protos[n] for n in names]);
    correct = 0; total = 0; conf = []
    for n in names:
        for ti in query_inits:
            if ti not in feats[n]: continue
            q = feats[n][ti]; sims = P @ q; pred = names[int(sims.argmax())]
            total += 1; correct += int(pred == n)
            if pred != n: conf.append((n, pred))
    print(f"\n=== DINOv2 visual SEPARABILITY of LIBERO objects (privileged crops, cross-init NN): "
          f"{correct}/{total} = {correct/max(total,1):.2f} ===", flush=True)
    print(f"  confusions (true->pred): {conf}", flush=True)
    print("DINO_SEP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
