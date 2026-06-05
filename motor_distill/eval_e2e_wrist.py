"""FULL pick+place with WRIST-CAMERA closed-loop binding refinement (principled: frozen foundation does
the fine binding; motor unchanged). Coarse third-person OWLv2 goal for approach; at grasp range, OWLv2 on
the WRIST image -> wrist depth + per-step wrist camera matrix -> refined cm-accurate 3D goal -> firm grasp.
Unified phase-conditioned motor. Reports reach / grasp / place + wrist_bind_err (close-range accuracy).
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_motor import head_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--motor", default="runs/motor_head.pkl"); ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxA", type=int, default=130); ap.add_argument("--maxB", type=int, default=110)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--place-cm", type=float, default=12.0)
    ap.add_argument("--near-cm", type=float, default=14.0)
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
    H = W = 256
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def locate(img, tgt):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(owlm(**inp), threshold=args.thr,
                                                            target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]; best = (float(sc.max()), int((b[1]+b[3])/2), int((b[0]+b[2])/2))
        return None if best is None else (best[1], best[2])

    def goal_from(cam, imgkey, depthkey, sim, obs, name, dz=0.0):
        img = np.asarray(obs[imgkey])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs[depthkey]).reshape(H, W, 1))[::-1].copy()
        rc = locate(img, name)
        if rc is None:
            return None
        cy, cx = rc; reg = rd[max(0, cy-6):cy+6, max(0, cx-6):cx+6, 0]
        d = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
        w2c = CU.get_camera_transform_matrix(sim, cam, H, W); c2w = np.linalg.inv(w2c)
        g = CU.transform_from_pixels_to_world(np.array([cy, cx], float), np.full((H, W, 1), d), c2w)[:3]
        return np.asarray(g, np.float32) + np.array([0, 0, dz], np.float32)

    def act(goal, obs, phase):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return np.array(head_apply(mp, jnp.asarray(ee - goal),
                                   jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                   jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32)),
                                   jnp.asarray([float(phase)])))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    reaches, lifts, succ, werr = [], [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; M = distractors[0] if distractors else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, [T, args.container + "_1"] + ([M] if M else [])); cb = rb[args.container + "_1"]
        except Exception:
            env.close(); continue
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; coarse = None; rT, rM = 1e9, 1e9; we = None
        for step in range(args.maxA):                                   # phase 0 approach + WRIST-refined grasp
            if coarse is None or step % 20 == 0:
                g = goal_from("agentview", "agentview_image", "agentview_depth", sim, obs, nm(T))
                if g is not None: coarse = g
            if coarse is None: break
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            goal = coarse
            if np.linalg.norm(ee - coarse) < args.near_cm / 100:        # close range -> wrist refine
                wg = goal_from("robot0_eye_in_hand", "robot0_eye_in_hand_image", "robot0_eye_in_hand_depth", sim, obs, nm(T))
                if wg is not None:
                    goal = wg
                    if we is None: we = float(np.linalg.norm(wg - body_pos(sim, rb[T])))
            obs, _, done, _ = env.step(act(goal, obs, 0).tolist())
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            rT = min(rT, float(np.linalg.norm(E - body_pos(sim, rb[T])))); rM = min(rM, float(np.linalg.norm(E - body_pos(sim, rb[M])))) if M else rM
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.04: break
        gB = None
        if lifted > 0.04:
            for step in range(args.maxB):                               # phase 1 transport + release
                if gB is None or step % 20 == 0:
                    g = goal_from("agentview", "agentview_image", "agentview_depth", sim, obs, args.container, dz=0.06)
                    if g is not None: gB = g
                if gB is None: break
                obs, _, done, _ = env.step(act(gB, obs, 1).tolist())
        mpz = body_pos(sim, rb[T]); cpz = body_pos(sim, cb)
        placed = bool(lifted > 0.04 and np.linalg.norm(mpz[:2] - cpz[:2]) < args.place_cm / 100)
        env.close()
        reaches.append((rT < rM) if M else (rT < 0.08)); lifts.append(lifted > 0.04); succ.append(placed)
        if we is not None: werr.append(we)
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={nm(T):13s} | wrist_err={'%.1fcm'%(we*100) if we else 'none':>7} "
              f"reached={reaches[-1]} lifted={lifted*100:.0f}cm placed={placed}", flush=True)
    n = len(succ)
    print(f"\n=== WRIST closed-loop (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  reach {sum(reaches)}/{n}={sum(reaches)/n*100:.0f}%  grasp {sum(lifts)}/{n}={sum(lifts)/n*100:.0f}%  "
          f"PLACE {sum(succ)}/{n}={sum(succ)/n*100:.0f}%  | wrist_bind_err {np.mean(werr)*100 if werr else float('nan'):.1f}cm (n={len(werr)})", flush=True)
    print("EVAL_E2E_WRIST_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
