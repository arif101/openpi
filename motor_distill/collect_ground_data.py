"""Collect LIBERO sim AUTO-LABELS for the learned grounding head (no manual labels; GT pixel from sim poses).
For each task x many inits: render agentview@res, cache FROZEN DINOv2 dense patch features (g x g x C), the target's
object PROTOTYPE (mean exemplar patch feature, open-vocab query), and the GT target patch coords (body_pos projection
-> patch grid). Position diversity across inits IS the augmentation that teaches generalization to LIBERO-PRO swaps.

The deployed head sees ONLY (dense feats, proto) -> predicts the target patch; body_pos is used ONLY to make the label.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python motor_distill/collect_ground_data.py --bddl-dir <obj_swap> --init-dir <obj_swap> --out data/ground_obj --res 448 --n 10
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from diag_dinodense import dino_dense


def target_from_stem(stem, objs):
    import re as _re
    m = _re.match(r"pick_up_the_(.+)$", stem)
    if not m: return None
    toks = m.group(1).split("_"); cut = len(toks)
    for s in ("between", "next", "on", "from", "in", "and"):
        if s in toks: cut = min(cut, toks.index(s))
    tc = "_".join(toks[:cut])
    return next((o for o in objs if tc and tc in o and "basket" not in o), None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--res", type=int, default=448)
    p.add_argument("--n", type=int, default=10); p.add_argument("--n-inits", type=int, default=40)
    p.add_argument("--proto-inits", default="42,44,46")   # exemplar inits for the proto (held out from train inits 0..n-inits)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outd = pathlib.Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"GROUND-DATA collect: res={args.res} n_inits={args.n_inits} on {pathlib.Path(args.bddl_dir).name}", flush=True)
    nsamp = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem; T = target_from_stem(stem, objs) or targets[0]
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        try: inits = np.asarray(torch.load(fi, weights_only=False))
        except Exception: continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=False)
        # ---- proto (open-vocab query): mean exemplar patch feature at the object over proto-inits ----
        protos = []
        for qi in proto_inits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, [T])
            if rb.get(T) is None: continue
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            w2p = cu.get_camera_transform_matrix(sim, "agentview", args.res, args.res)
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, args.res, args.res)[0]
            r_up = int(args.res - 1 - px[0]); c = int(px[1])
            fg, g = dino_dense(hi, dev, args.res)
            pr = int(r_up / args.res * g); pc = int(c / args.res * g)
            patch = fg[max(0, pr-1):pr+2, max(0, pc-1):pc+2].reshape(-1, fg.shape[-1])
            if len(patch): protos.append(patch.mean(0))
        if not protos: env.close(); continue
        proto = np.mean(protos, 0); proto = (proto / (np.linalg.norm(proto) + 1e-6)).astype(np.float16)
        # ---- training samples: dense feats + GT patch over n_inits ----
        feats = []; gts = []; G = None
        nt = min(args.n_inits, len(inits))
        for ti in range(nt):
            env.seed(ti); env.reset(); env.set_init_state(inits[ti]); sim = env.env.sim
            rb = resolve_bodies(sim, [T])
            if rb.get(T) is None: continue
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, args.res); G = g
            w2p = cu.get_camera_transform_matrix(sim, "agentview", args.res, args.res)
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, args.res, args.res)[0]
            r_up = args.res - 1 - px[0]; c = px[1]
            gr = int(np.clip(r_up / args.res * g, 0, g - 1)); gc = int(np.clip(c / args.res * g, 0, g - 1))
            feats.append(fg.astype(np.float16)); gts.append([gr, gc])
        if feats:
            np.savez_compressed(outd / f"{stem[:40]}.npz",
                                feats=np.stack(feats), proto=proto, gt=np.array(gts, np.int64), g=G,
                                noun=str(T))
            nsamp += len(feats)
            print(f"  {stem[:34]:36s} {len(feats)} samples (g={G})", flush=True)
        env.close()
    print(f"\n=== GROUND-DATA: {nsamp} samples -> {args.out} ===", flush=True)
    print("GROUNDDATA_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
