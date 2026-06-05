"""Calibrate the WRIST-camera unprojection (brute-force flip/axis convention vs ground truth) and check
whether OWLv2 detects the target in the wrist view at close range. Decides if wrist refinement is viable."""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_motor import head_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--motor", default="runs/motor_head.pkl"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--approach", type=int, default=60)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    mp = jax.tree.map(jnp.asarray, pickle.load(open(args.motor, "rb")))
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    WC = "robot0_eye_in_hand"

    @torch.no_grad()
    def detect(img, tgt):
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(owlm(**inp), threshold=0.01,
                                                            target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc):
                b = r["boxes"].cpu().numpy()[sc.argmax()]; return int((b[1]+b[3])/2), int((b[0]+b[2])/2)
        return None

    CONV = [("vflip", lambda d: d[::-1].copy(), lambda r: (H-1-r[0], r[1])),
            ("vraw", lambda d: d, lambda r: (r[0], r[1])),
            ("hflip", lambda d: d[:, ::-1].copy(), lambda r: (r[0], W-1-r[1])),
            ("vhflip", lambda d: d[::-1, ::-1].copy(), lambda r: (H-1-r[0], W-1-r[1]))]

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    votes = {}; det = 0; tot = 0
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); tp = body_pos(sim, rb[T]).astype(np.float64)
        # drive EE toward the true object (oracle goal) to get close for the wrist view
        for _ in range(args.approach):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            a = np.array(head_apply(mp, jnp.asarray(ee - tp.astype(np.float32)),
                                    jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                    jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32)),
                                    jnp.asarray([0.0])))
            obs, _, done, _ = env.step(a.tolist())
            if np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - tp) < 0.08:
                break
        dist = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - tp))
        rd = CU.get_real_depth_map(sim, np.asarray(obs[WC + "_depth"]).reshape(H, W, 1))
        w2c = CU.get_camera_transform_matrix(sim, WC, H, W); c2w = np.linalg.inv(w2c)
        proj = np.asarray(CU.project_points_from_world_to_camera(tp[None], w2c, H, W)[0])
        pr, pc = int(np.clip(proj[0], 0, H-1)), int(np.clip(proj[1], 0, W-1))
        best = None
        for name, dflip, order in CONV:
            rr, cc = order((pr, pc)); rr, cc = int(np.clip(rr, 0, H-1)), int(np.clip(cc, 0, W-1))
            err = float(np.linalg.norm(CU.transform_from_pixels_to_world(np.array([rr, cc], float), dflip(rd), c2w)[:3] - tp))
            if best is None or err < best[1]: best = (name, err)
        votes[best[0]] = votes.get(best[0], 0) + 1
        # OWLv2 detect in wrist view (try vflip frame)
        wimg = np.asarray(obs[WC + "_image"])[::-1].copy()
        d = detect(wimg, nm(T)); det += (d is not None); tot += 1
        env.close()
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={nm(T):13s} ee_dist={dist*100:.0f}cm | geom={best[0]}:{best[1]*100:.1f}cm | owl_wrist={'yes' if d else 'NO'}", flush=True)
    print(f"\n=== WRIST CALIBRATION (N={tot}) ===", flush=True)
    print(f"  convention votes: {votes}   OWLv2-detects-in-wrist: {det}/{tot}", flush=True)
    print("WRIST_CHECK_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
