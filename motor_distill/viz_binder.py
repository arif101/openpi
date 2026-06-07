"""Visualize the binder: render @res, draw ALL proposal boxes + the CHOSEN box (red) + the GT object box
(green), crop both the chosen and the GT region, and print SigLIP scores (top-k names) on EACH crop.
Reveals whether failures are bad CROPS (proposal) or bad RECOGNITION. Saves annotated PNGs + a crops panel.

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl .venv/bin/python motor_distill/viz_binder.py \
       --tasks milk,tomato_sauce,cream_cheese --res 1024 --out viz
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from PIL import Image, ImageDraw
from cf_harness import parse_bddl, resolve_bodies, body_pos
from bind_foveate import detect_all, clip_scores, nm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-dir", default="/root/LIBERO-PRO/libero/libero/bddl_files/libero_object")
    p.add_argument("--init-dir", default="/root/LIBERO-PRO/libero/libero/init_files/libero_object")
    p.add_argument("--tasks", default="milk,tomato_sauce,cream_cheese")
    p.add_argument("--res", type=int, default=1024); p.add_argument("--init", type=int, default=20)
    p.add_argument("--out", default="viz")
    args = p.parse_args()
    import robosuite.utils.camera_utils as cu
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(args.out); out.mkdir(exist_ok=True); R = args.res
    want_tasks = args.tasks.split(",")
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    for bf in bddls:
        stem = pathlib.Path(bf).stem
        if not any(w in stem for w in want_tasks): continue
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; phrases = [nm(o) for o in objs]
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{stem}.pruned_init", weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=True)
        env.seed(args.init); env.reset(); obs = env.set_init_state(inits[args.init]); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); true3d = body_pos(sim, rb[T])
        rgb = np.asarray(obs["agentview_image"]); up = rgb[::-1]                          # upright
        # GT object pixel (native) -> upright row
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        gp = cu.project_points_from_world_to_camera(true3d[None], w2p, R, R)[0]            # [row,col] native
        gt_r_up = R - 1 - gp[0]; gt_c = gp[1]
        # proposals + chosen
        boxes = detect_all(up, " . ".join(phrases) + " .", dev) or detect_all(up, "an object .", dev)
        crops, keep = [], []
        for b in boxes:
            x0, y0, x1, y1 = b
            if x1 - x0 < 4 or y1 - y0 < 4: continue
            crops.append(up[y0:y1, x0:x1]); keep.append(b)
        S = clip_scores(crops, phrases, dev); ti = phrases.index(nm(T))
        assigned = S.argmax(1); cand = [i for i in range(len(keep)) if assigned[i] == ti]
        bi = (max(cand, key=lambda i: S[i, ti]) if cand else int(S[:, ti].argmax()))
        cb = keep[bi]
        # draw
        im = Image.fromarray(up.copy()); d = ImageDraw.Draw(im)
        for b in keep: d.rectangle([int(b[0]), int(b[1]), int(b[2]), int(b[3])], outline=(255, 200, 0), width=1)
        d.rectangle([int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])], outline=(255, 0, 0), width=4)   # chosen
        d.ellipse([gt_c - 9, gt_r_up - 9, gt_c + 9, gt_r_up + 9], outline=(0, 255, 0), width=4)        # GT
        im.save(out / f"{T}_scene.png")
        # crops + scores
        ch_crop = up[cb[1]:cb[3], cb[0]:cb[2]]
        gy0, gy1 = int(max(0, gt_r_up - 60)), int(min(R, gt_r_up + 60)); gx0, gx1 = int(max(0, gt_c - 60)), int(min(R, gt_c + 60))
        gt_crop = up[gy0:gy1, gx0:gx1]
        Image.fromarray(ch_crop).save(out / f"{T}_chosen_crop.png")
        Image.fromarray(gt_crop).save(out / f"{T}_gt_crop.png")
        Sc = clip_scores([ch_crop, gt_crop], phrases, dev)
        def top(row):
            o = np.argsort(-row)[:3]; return ", ".join(f"{phrases[j]}={row[j]:.1f}" for j in o)
        print(f"\n### {T}  (want='{nm(T)}', {len(keep)} boxes)", flush=True)
        print(f"  CHOSEN crop top3: {top(Sc[0])}", flush=True)
        print(f"  GT     crop top3: {top(Sc[1])}", flush=True)
        env.close()
    print("VIZ_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
