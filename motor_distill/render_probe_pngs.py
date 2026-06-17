"""Render agentview PNGs + GT target pixels for the LocateAnything isolated probe (system python; sim here).
The separate laenv (transformers 4.57.1 + LocateAnything) reads these PNGs and points at the noun -> bind accuracy.
GT pixel (body_pos projection) is used ONLY for scoring. Saves <out>/<stem>_<init>.png + <out>/labels.json.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json \
  python motor_distill/render_probe_pngs.py --bddl-dir <obj_swap> --init-dir <obj_swap> --out data/la_pngs --res 448 --n 10
"""
from __future__ import annotations
import argparse, glob, pathlib, json
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos
from collect_ground_data import target_from_stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", required=True); p.add_argument("--init-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--res", type=int, default=448)
    p.add_argument("--n", type=int, default=10); p.add_argument("--query-inits", default="20,22,24,26,28")
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from PIL import Image
    outd = pathlib.Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    qinits = [int(x) for x in args.query_inits.split(",")]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    labels = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem; T = target_from_stem(stem, objs) or targets[0]
        from bind_foveate import nm
        noun = nm(T).replace("_", " ")
        fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
        if not fi.exists(): continue
        try: inits = np.asarray(torch.load(fi, weights_only=False))
        except Exception: continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256, camera_depths=False)
        for qi in qinits:
            if qi >= len(inits): continue
            env.seed(qi); env.reset(); env.set_init_state(inits[qi]); sim = env.env.sim
            rb = resolve_bodies(sim, [T])
            if rb.get(T) is None: continue
            hi = np.asarray(sim.render(width=args.res, height=args.res, camera_name="agentview"))[::-1].copy()  # upright
            w2p = cu.get_camera_transform_matrix(sim, "agentview", args.res, args.res)
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, args.res, args.res)[0]
            gt_r = float(args.res - 1 - px[0]); gt_c = float(px[1])     # upright pixel (row,col) at res
            fn = f"{stem[:36]}_{qi}.png"
            Image.fromarray(hi).save(outd / fn)
            labels.append({"png": fn, "noun": noun, "gt_r": gt_r, "gt_c": gt_c, "res": args.res})
        env.close()
        print(f"  {stem[:34]:36s} rendered", flush=True)
    json.dump(labels, open(outd / "labels.json", "w"))
    print(f"=== rendered {len(labels)} probe images -> {args.out} ===", flush=True)
    print("RENDER_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
