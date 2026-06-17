"""CATALOG-FREE foveated identity via CLIP zero-shot on hi-res crops (tests the 'zoom + open-vocab classifier' idea, no
reference catalog). For each scene: tight hi-res crop of each object (oracle box to isolate the CLASSIFIER from proposal
noise) -> CLIP image emb; CLIP text emb of each candidate name. Metric: given the TARGET name, does argmax-over-crops of
CLIP(crop, 'a photo of <name>') pick the TRUE target? (= catalog-free identity, comparable to DINOv2 catalog 90%).

Run: PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv/bin/python motor_distill/diag_clip_crops.py \
       --bddl-dir <swap> --init-dir <swap> --n 10 --init-start 20 --hr 1024 --crop 70
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", required=True); ap.add_argument("--init-dir", default="")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--hr", type=int, default=1024); ap.add_argument("--crop", type=int, default=70)
    ap.add_argument("--model", default="openai/clip-vit-large-patch14"); ap.add_argument("--container", default="basket")
    args = ap.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = "cuda"; R = args.hr
    clip = CLIPModel.from_pretrained(args.model).to(dev).eval(); proc = CLIPProcessor.from_pretrained(args.model)
    # global vocabulary = union of all object names across the suite (open-vocab text labels, NO image references)
    allnames = set()
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))
    for bf in bddls:
        _, objs, _, _ = parse_bddl(bf)
        for o in objs:
            if args.container not in o: allnames.add(o.rsplit("_", 1)[0].replace("_", " "))
    vocab = sorted(allnames)
    txt = proc(text=[f"a photo of a {n}" for n in vocab], return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():
        tfeat = clip.get_text_features(**txt); tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    print(f"vocab ({len(vocab)}): {vocab}", flush=True)

    tgt_hit = []; crop_acc = []
    for bf in bddls[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; tname = T.rsplit("_", 1)[0].replace("_", " ")
        inits = np.asarray(torch.load(pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init", weights_only=False))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R, camera_depths=False)
        env.seed(args.init_start); env.reset(); sim = env.env.sim; env.set_init_state(inits[args.init_start])
        graspables = [o for o in objs if args.container not in o]; rb = resolve_bodies(sim, graspables)
        w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        img_up = np.asarray(sim.render(width=R, height=R, camera_name="agentview"))[::-1].copy()
        crops = []; names = []
        for o in graspables:
            if rb.get(o) is None: continue
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            cy = int(R - 1 - px[0]); cx = int(px[1]); h = args.crop
            cr = img_up[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
            if cr.size < 100: continue
            crops.append(Image.fromarray(cr)); names.append(o.rsplit("_", 1)[0].replace("_", " "))
        if not crops: env.close(); continue
        im = proc(images=crops, return_tensors="pt").to(dev)
        with torch.no_grad():
            ifeat = clip.get_image_features(**im); ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
        sims = (ifeat @ tfeat.T).cpu().numpy()       # [n_crops, vocab]
        # per-crop label = argmax vocab; crop_acc = does each crop classify to its own name
        for j, nm_j in enumerate(names):
            crop_acc.append(int(vocab[int(sims[j].argmax())] == nm_j))
        # BINDER metric: given target name, pick the crop with max CLIP(crop, target) -> is it the true target crop?
        ti = vocab.index(tname); jbest = int(sims[:, ti].argmax())
        hit = int(names[jbest] == tname); tgt_hit.append(hit)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} target='{tname}' picked='{names[jbest]}' hit={hit}", flush=True)
        env.close()
    print(f"\n=== CLIP catalog-FREE foveated identity, {pathlib.Path(args.bddl_dir).name} N={len(tgt_hit)} hr={R} crop={args.crop} ===", flush=True)
    print(f"  TARGET binder hit-rate = {np.mean(tgt_hit):.2f}   (DINOv2 catalog = 0.90)", flush=True)
    print(f"  per-crop name accuracy = {np.mean(crop_acc):.2f} (n={len(crop_acc)})", flush=True)
    print("CLIPCROP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
