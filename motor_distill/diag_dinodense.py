"""ZERO-SHOT DINOv2 DENSE-CORRESPONDENCE localization probe (no body_pos in the prediction).
Tests whether dense patch-feature matching localizes LIBERO's ~20px objects better than SAM-region centroids
(the off-the-shelf ceiling = 21/30). If dense correspondence is strong, a small learned head on top will beat it.

Per task: build a PATCH-SPACE object prototype from a few exemplar inits (privileged crop = reference template only,
like the existing proto bank), then on QUERY inits run DINOv2 dense on the FULL scene -> per-patch cosine-sim to the
prototype -> argmax patch -> predicted pixel. Score = |pred - GT pixel| < THRESH (GT = body_pos projection, used ONLY
for scoring, never for the prediction).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
     python motor_distill/diag_dinodense.py --bddl-dir <object_swap> --init-dir <object_swap> --n 10 --res 448
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos

_M = {}


def dino_dense(img_rgb, dev, res=448, mid="facebook/dinov2-large"):
    """Full-image -> (gh, gw, C) L2-normalized patch-token grid. img_rgb = HxWx3 uint8 (upright)."""
    from PIL import Image
    if "m" not in _M:
        from transformers import AutoModel, AutoImageProcessor
        _M["m"] = AutoModel.from_pretrained(mid).to(dev).eval()
        _M["p"] = AutoImageProcessor.from_pretrained(mid)
    model, proc = _M["m"], _M["p"]
    inp = proc(images=Image.fromarray(img_rgb), return_tensors="pt", size={"height": res, "width": res}).to(dev)
    with torch.no_grad():
        out = model(**inp)
    tok = out.last_hidden_state[0, 1:]                 # drop CLS -> (P, C)  (dinov2-large: no register tokens)
    g = int(round(tok.shape[0] ** 0.5))
    f = tok.reshape(g, g, -1)
    f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return f.cpu().numpy(), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--n", type=int, default=10); p.add_argument("--res", type=int, default=448)
    p.add_argument("--proto-inits", default="30,32,34"); p.add_argument("--query-inits", default="20,22,24,26,28")
    p.add_argument("--thresh", type=int, default=18)   # pixel tolerance in the 256 projection frame (matches eval bind_ok)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = 256; HALF = 60
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    query_inits = [int(x) for x in args.query_inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"DINOv2-DENSE localization probe: res={args.res} thresh={args.thresh}px on {pathlib.Path(args.bddl_dir).name}", flush=True)
    hit = 0; tot = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem
        import re as _re
        m = _re.match(r"pick_up_the_(.+)$", stem); T = targets[0]
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for s in ("between", "next", "on", "from", "in", "and"):
                if s in toks: cut = min(cut, toks.index(s))
            tc = "_".join(toks[:cut]); cand = next((o for o in objs if tc and tc in o and "basket" not in o), None)
            if cand: T = cand
        inits = None
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if fi.exists():
            try: inits = np.asarray(torch.load(fi, weights_only=False))
            except Exception: inits = None
        if inits is None: continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        # ---- build PATCH-SPACE prototype from exemplar inits (privileged crop = reference template ONLY) ----
        protos = []
        for qi in proto_inits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, [T]);
            if rb.get(T) is None: continue
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            w2p = cu.get_camera_transform_matrix(sim, "agentview", args.res, args.res)
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, args.res, args.res)[0]
            r_up = int(args.res - 1 - px[0]); c = int(px[1])
            fg, g = dino_dense(hi, dev, args.res)
            pr = int(r_up / args.res * g); pc = int(c / args.res * g)
            patch = fg[max(0, pr-1):pr+2, max(0, pc-1):pc+2].reshape(-1, fg.shape[-1])   # 3x3 patch nbhd at the object
            if len(patch): protos.append(patch.mean(0))
        if not protos: env.close(); continue
        proto = np.mean(protos, 0); proto = proto / (np.linalg.norm(proto) + 1e-6)
        # ---- QUERY: dense match on full scene, NO body_pos in prediction ----
        for qi in query_inits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, [T])
            if rb.get(T) is None: continue
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, args.res)
            simmap = fg @ proto                                  # (g,g) cosine sim
            pr, pc = np.unravel_index(int(simmap.argmax()), simmap.shape)
            # patch -> hi-res upright pixel -> 256 projection frame
            r_up = (pr + 0.5) / g * args.res; c = (pc + 0.5) / g * args.res
            pred_r256 = R - 1 - r_up / (args.res / R); pred_c256 = c / (args.res / R)
            # GT pixel (scoring only)
            w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
            gt = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], R, R)[0] if False else \
                 cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, R, R)[0]
            d = float(np.hypot(pred_r256 - gt[0], pred_c256 - gt[1]))
            hit += int(d < args.thresh); tot += 1
        env.close()
        print(f"  {stem[:34]:36s} running {hit}/{tot}", flush=True)
    print(f"\n=== DINOv2-DENSE localization: {hit}/{tot} = {hit/max(tot,1):.3f} within {args.thresh}px (vs SAM+DINO 21/30=0.70) ===", flush=True)
    print("DINODENSE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
