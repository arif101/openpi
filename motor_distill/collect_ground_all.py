"""ALL-OBJECT grounding data for the learned identity-localizer head (extends collect_ground_data to every scene object,
not just the parsed target). Per task: render agentview@res over inits, cache FROZEN DINOv2 dense feats ONCE per init,
and for EVERY resolvable scene object record its GT target patch + build its patch-space proto. -> the head learns to
localize ANY named object by its prototype; the relational which-instance stage (spatial) then picks among predicted
peaks by geometry (no body_pos at deploy). Sim GT poses used ONLY to make labels.

npz per task: feats(n,g,g,C fp16), objs[list], gt(n_obj,n,2 int), protos(n_obj,C fp16), nouns[list], g.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python motor_distill/collect_ground_all.py --bddl-dir <spatial_swap> --init-dir <spatial_swap> --out data/ground_spatial --res 448 --n 10
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from diag_dinodense import dino_dense


def patch_at(sim, body, res, g, fg, cu):
    w2p = cu.get_camera_transform_matrix(sim, "agentview", res, res)
    px = cu.project_points_from_world_to_camera(body_pos(sim, body)[None], w2p, res, res)[0]
    r_up = res - 1 - px[0]; c = px[1]
    gr = int(np.clip(r_up / res * g, 0, g - 1)); gc = int(np.clip(c / res * g, 0, g - 1))
    return gr, gc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--res", type=int, default=448)
    p.add_argument("--n", type=int, default=10); p.add_argument("--n-inits", type=int, default=40)
    p.add_argument("--proto-inits", default="42,44,46")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from eval_physics_place import scene_bodies
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outd = pathlib.Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    proto_inits = [int(x) for x in args.proto_inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"GROUND-ALL collect: res={args.res} n_inits={args.n_inits} on {pathlib.Path(args.bddl_dir).name}", flush=True)
    nsamp = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        stem = pathlib.Path(bf).stem
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        try: inits = np.asarray(torch.load(fi, weights_only=False))
        except Exception: continue
        cand = scene_bodies(bf)                                   # all object+fixture instances in the scene
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=False)
        # resolve which candidates exist as bodies (use first init)
        env.seed(0); env.reset(); env.set_init_state(inits[0]); sim = env.env.sim
        rb0 = resolve_bodies(sim, cand); present = [o for o in cand if rb0.get(o) is not None]
        if not present: env.close(); continue
        # ---- protos per object (exemplar patch feature) ----
        protos = {}
        for qi in proto_inits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, present)
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, args.res)
            for o in present:
                if rb.get(o) is None: continue
                gr, gc = patch_at(sim, rb[o], args.res, g, fg, cu)
                patch = fg[max(0, gr-1):gr+2, max(0, gc-1):gc+2].reshape(-1, fg.shape[-1])
                if len(patch): protos.setdefault(o, []).append(patch.mean(0))
        protos = {o: (np.mean(v, 0) / (np.linalg.norm(np.mean(v, 0)) + 1e-6)).astype(np.float16) for o, v in protos.items() if v}
        objs_k = [o for o in present if o in protos]
        if not objs_k: env.close(); continue
        # ---- feats per init + GT patch per object ----
        feats = []; gts = [[] for _ in objs_k]; G = None
        nt = min(args.n_inits, len(inits))
        for ti in range(nt):
            env.seed(ti); env.reset(); env.set_init_state(inits[ti]); sim = env.env.sim
            rb = resolve_bodies(sim, objs_k)
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()
            fg, g = dino_dense(hi, dev, args.res); G = g
            feats.append(fg.astype(np.float16))
            for j, o in enumerate(objs_k):
                if rb.get(o) is None: gts[j].append([-1, -1]); continue
                gts[j].append(list(patch_at(sim, rb[o], args.res, g, fg, cu)))
        np.savez_compressed(outd / f"{stem[:40]}.npz",
                            feats=np.stack(feats), objs=np.array(objs_k),
                            gt=np.array(gts, np.int64), protos=np.stack([protos[o] for o in objs_k]),
                            nouns=np.array([o for o in objs_k]), g=G)
        nsamp += len(feats) * len(objs_k)
        print(f"  {stem[:34]:36s} {len(feats)} inits x {len(objs_k)} objs (g={G})", flush=True)
        env.close()
    print(f"\n=== GROUND-ALL: {nsamp} obj-samples -> {args.out} ===", flush=True)
    print("GROUNDALL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
