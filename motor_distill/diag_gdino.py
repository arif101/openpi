"""Catalog-FREE open-vocab grounding test: GroundingDINO maps the language object-name -> box, zero-shot, NO prototype
catalog, NO reference positions. For each swap scene, query the target name (and 'basket') on the agentview and measure
(a) identity: does the top box land on the true object? (b) localization: box-center pixel vs true projection. If solid,
this replaces the DINOv2 catalog entirely (truly from images + language). Depth then lifts the box-center to 3D.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/diag_gdino.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--box-thr", type=float, default=0.25); ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--container", default="basket"); ap.add_argument("--hr", type=int, default=0)  # render agentview at hr for grounding (foveation)
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image
    dev = "cuda"; R = args.res
    proc = AutoProcessor.from_pretrained(args.model)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(dev).eval()

    def ground(img_up, phrase):
        """Return top box center (row,col) in matrix-native 256 coords + score, for `phrase`, or None. img_up may be HR."""
        H = img_up.shape[0]; scale = R / H
        pil = Image.fromarray(img_up)
        inp = proc(images=pil, text=phrase.lower().strip() + ".", return_tensors="pt").to(dev)
        with torch.no_grad(): out = gd(**inp)
        res = proc.post_process_grounded_object_detection(out, inp["input_ids"], threshold=args.box_thr,
                                                          text_threshold=args.text_thr, target_sizes=[pil.size[::-1]])[0]
        if len(res["boxes"]) == 0: return None
        i = int(torch.argmax(res["scores"])); b = res["boxes"][i].cpu().numpy()  # x0,y0,x1,y1 in upright HR px
        cx = (b[0] + b[2]) / 2 * scale; cy_up = (b[1] + b[3]) / 2 * scale         # -> 256 upright
        return (R - 1 - cy_up, cx, float(res["scores"][i]))   # matrix-native row, col (256), score

    tgt_id = []; tgt_px = []; con_id = []; con_px = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        inits = np.asarray(torch.load(fi, weights_only=False)) if fi.exists() else None
        ti = args.init_start
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(ti); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
        rb = resolve_bodies(sim, [T, args.container + "_1"])
        if args.hr:
            img_up = np.asarray(sim.render(width=args.hr, height=args.hr, camera_name="agentview"))[::-1].copy()
        else:
            img_up = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1])
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        phrase = T.rsplit("_", 1)[0].replace("_", " ")          # e.g. "alphabet_soup_1" -> "alphabet soup"
        gt = ground(img_up, phrase)
        if rb.get(T) is not None and gt is not None:
            pxt = cu.project_points_from_world_to_camera(body_pos(sim, rb[T])[None], w2p, R, R)[0]
            err = float(np.hypot(gt[0] - pxt[0], gt[1] - pxt[1])); tgt_px.append(err); tgt_id.append(int(err < 22))
            print(f"  {pathlib.Path(bf).stem[:22]:24s} '{phrase:14s}' box_score={gt[2]:.2f} px_err={err:4.1f}  hit={err<22}", flush=True)
        else:
            tgt_id.append(0); print(f"  {pathlib.Path(bf).stem[:22]:24s} '{phrase:14s}' NO BOX", flush=True)
        gc = ground(img_up, args.container)
        cb = rb.get(args.container + "_1")
        if cb is None:
            cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]; cb = cand[0] if cand else None
        if cb is not None and gc is not None:
            pxc = cu.project_points_from_world_to_camera(body_pos(sim, cb)[None], w2p, R, R)[0]
            ec = float(np.hypot(gc[0] - pxc[0], gc[1] - pxc[1])); con_px.append(ec); con_id.append(int(ec < 30))
        else: con_id.append(0)
        env.close()
    print(f"\n=== GroundingDINO open-vocab (catalog-FREE) on {pathlib.Path(args.bddl_dir).name} N={len(tgt_id)} ===", flush=True)
    print(f"  TARGET  identity(hit<22px)={np.mean(tgt_id):.2f}  px_err={np.mean(tgt_px):.1f} (n={len(tgt_px)})", flush=True)
    print(f"  BASKET  identity(hit<30px)={np.mean(con_id):.2f}  px_err={np.mean(con_px) if con_px else 0:.1f} (n={len(con_px)})", flush=True)
    print("GDINO_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
