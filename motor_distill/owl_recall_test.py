"""De-hardcoding feasibility: does OWLv2 DETECT the graspable LIBERO objects (no body_pos)? For each object-suite
scene, run OWLv2 with the object names, and check whether a detected box lands near each object's true projected
center. Reports detection recall + box-center error. If recall is high, we can replace privileged body_pos proposals
with OWLv2 -> fully honest perception.
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="/root/LIBERO-Pro-data/bddl_files/libero_object_swap")
    ap.add_argument("--init-dir", default="/root/LIBERO-Pro-data/init_files/libero_object_swap")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--init-start", type=int, default=20)
    ap.add_argument("--res", type=int, default=256); ap.add_argument("--thr", type=float, default=0.05)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as cu
    from PIL import Image
    from cf_harness import parse_bddl, resolve_bodies, body_pos
    dev = "cuda"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    R = args.res; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    hit = 0; tot = 0; errs = []
    for bf in sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        graspables = [o for o in objs if "basket" not in o]
        fi = pathlib.Path(args.init_dir) / f"{pathlib.Path(bf).stem}.pruned_init"
        import torch as T
        inits = np.asarray(T.load(fi, weights_only=False)) if fi.exists() else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=R, camera_widths=R)
        env.seed(args.init_start); env.reset(); sim = env.env.sim
        obs = env.set_init_state(inits[args.init_start]) if inits is not None else env.reset()
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        names = list(dict.fromkeys(nm(o) for o in graspables))
        boxes = []
        with torch.no_grad():
            for pr in [[f"a {q}" for q in names]]:
                inp = owlp(text=pr, images=Image.fromarray(img), return_tensors="pt").to(dev)
                r = owlp.post_process_grounded_object_detection(owlm(**inp), threshold=args.thr,
                                                                target_sizes=torch.tensor([[R, R]]).to(dev))[0]
                for b in r["boxes"].cpu().numpy(): boxes.append([(b[1] + b[3]) / 2, (b[0] + b[2]) / 2])  # (row,col) center
        rb = resolve_bodies(sim, graspables); w2p = cu.get_camera_transform_matrix(sim, "agentview", R, R)
        for o in graspables:
            if rb.get(o) is None: continue
            tot += 1
            px = cu.project_points_from_world_to_camera(body_pos(sim, rb[o])[None], w2p, R, R)[0]
            tr = R - 1 - px[0]; tc = px[1]   # account for the vertical flip applied to img
            if boxes:
                d = min(np.hypot(b[0] - tr, b[1] - tc) for b in boxes)
                if d < 20: hit += 1; errs.append(float(d))
        print(f"  {pathlib.Path(bf).stem[:26]:28s} boxes={len(boxes)} graspables={len([o for o in graspables if rb.get(o)])}", flush=True)
        env.close()
    print(f"\n=== OWLv2 detection recall: {hit}/{tot} = {100*hit/max(tot,1):.0f}%  box-center err(px) mean={np.mean(errs) if errs else -1:.1f} ===", flush=True)
    print("OWLREC_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
