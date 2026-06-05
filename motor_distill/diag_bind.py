"""Binder precision diagnostic: decompose the named-target localization error into PIXEL (detection)
vs DEPTH error, against ground truth. No rollouts -- localize the initial frame and compare to the true
object projected into the image. Decides the binder fix: better detector/mask (pixel) vs better depth.

Per target T (the NAMED counterfactual target):
  owl_px        OWLv2 box-center pixel (row,col)
  true_px       true object 3D -> projected pixel
  px_err        |owl_px - true_px| in pixels
  owl_depth     median depth in patch at owl box center
  true_depth    true object distance from camera
  bind_err3d    ||unproject(owl_px, owl_depth) - true_obj_xyz||      (full 3D binding error)
  bind_err_pxonly ||unproject(true_px? no) -> use owl_px with TRUE depth|| isolates pixel contribution

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/diag_bind.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def locate(img, tgt):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]
                best = (float(sc.max()), int((b[1]+b[3])/2), int((b[0]+b[2])/2))
        return None if best is None else (best[1], best[2])

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    PXE, B3, BPX, BDP = [], [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); obj_xyz = body_pos(sim, rb[T]).astype(np.float32)
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
        # true object projected pixel (account for the vertical flip used on img/rd)
        tp = CU.project_points_from_world_to_camera(obj_xyz[None], w2c, H, W)[0]
        true_r, true_c = int(H - 1 - tp[0]), int(tp[1])           # vflip row
        rc = locate(img, nm(T))
        if rc is None:
            env.close(); print(f"  {nm(T):15s} NO DETECTION"); continue
        oy, ox = rc
        patch = lambda r, c: float(np.median(rd[max(0,r-6):r+6, max(0,c-6):c+6, 0]))
        owl_d = patch(oy, ox); true_d = patch(true_r, true_c)
        unproj = lambda r, c, d: np.asarray(CU.transform_from_pixels_to_world(
            np.array([r, c], float), np.full((H, W, 1), d), c2w)[:3], np.float32)
        g_owl = unproj(oy, ox, owl_d)                            # what the binder actually returns
        g_pxonly = unproj(oy, ox, true_d)                        # owl pixel + TRUE depth -> isolates pixel error
        px_err = float(np.hypot(oy - true_r, ox - true_c))
        b3 = float(np.linalg.norm(g_owl[:2] - obj_xyz[:2]))
        bpx = float(np.linalg.norm(g_pxonly[:2] - obj_xyz[:2]))
        bdp = abs(owl_d - true_d)
        PXE.append(px_err); B3.append(b3); BPX.append(bpx); BDP.append(bdp)
        env.close()
        print(f"  {nm(T):15s} px_err={px_err:4.0f}px  owl_d={owl_d:.2f} true_d={true_d:.2f} depth_err={bdp*100:3.0f}cm "
              f"| bind3d={b3*100:4.0f}cm  (pixel-only={bpx*100:4.0f}cm)", flush=True)
    n = len(B3)
    print(f"\n=== BINDER PRECISION (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  mean 3D bind err   : {np.mean(B3)*100:.0f}cm   (median {np.median(B3)*100:.0f}cm)", flush=True)
    print(f"  from PIXEL error   : {np.mean(BPX)*100:.0f}cm   (owl pixel + true depth)", flush=True)
    print(f"  mean depth err     : {np.mean(BDP)*100:.0f}cm   mean pixel err {np.mean(PXE):.0f}px", flush=True)
    print(f"  -> dominant cause  : {'PIXEL/detection' if np.mean(BPX) > np.mean(BDP) else 'DEPTH'}", flush=True)
    print("DIAG_BIND_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
