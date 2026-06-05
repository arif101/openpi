"""Grasp root-cause diagnostic. The reach head grasps firmly in demos (21-31cm lifts) but slips in
closed loop (4-6cm marginal). WHY? Classify each grasp attempt by the geometry at gripper-closure:

  XY_OFFSET    at closure, EE horizontal dist to object > xy_tol  (closing beside the object)
  TOO_HIGH     at closure, EE is > z_tol above the object top      (closing above it)
  TOO_LOW      at closure, EE is below the object base             (pushing it away first)
  TIMING       closes before reaching the object / never settles
  FIRM         closed well-aligned AND object lifted >10cm sustained (a real grasp)

Logs, at the moment the gripper command first goes closed, the EE-vs-object offset (xy, dz) and the
resulting sustained lift. Tells us if the fix is approach-xy, closure-height, or genuine servo (P2).

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/diag_grasp.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxA", type=int, default=130); ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--cam", default="agentview"); ap.add_argument("--xy-tol", type=float, default=0.03)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach_head, "rb")))
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
                best = (float(sc.max()), int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2))
        return None if best is None else (best[1], best[2])

    def to_goal(img, rd, c2w, name):
        rc = locate(img, name)
        if rc is None: return None
        cy, cx = rc; reg = rd[max(0, cy-6):cy+6, max(0, cx-6):cx+6, 0]
        d = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                            np.full((H, W, 1), d), c2w)[:3], np.float32)

    def act(goal, obs):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return np.array(head_apply(rp, jnp.asarray(ee - goal),
                                   jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                   jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    classes = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        rb = resolve_bodies(sim, [T]); z0 = body_pos(sim, rb[T])[2]
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        goalT = None; closure = None; lifted = 0.0; prev_grip = -1.0
        for step in range(args.maxA):
            if goalT is None or step % 20 == 0:
                img = np.asarray(obs["agentview_image"])[::-1].copy()
                rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
                g = to_goal(img, rd, c2w, nm(T))
                if g is not None: goalT = g
            if goalT is None: break
            a = act(goalT, obs)
            ee = np.asarray(obs["robot0_eef_pos"], np.float32); op = body_pos(sim, rb[T])
            if closure is None and a[6] > 0.5 and prev_grip <= 0.5:   # first closure event
                gerr = float(np.linalg.norm(goalT[:2] - op[:2]))       # localized-goal vs TRUE object (binding error)
                eeg = float(np.linalg.norm(ee[:2] - goalT[:2]))        # EE vs its goal (motor tracking error)
                closure = dict(xy=float(np.linalg.norm(ee[:2] - op[:2])), dz=float(ee[2] - op[2]),
                               obj_top=float(op[2] - z0), gerr=gerr, eeg=eeg)
            prev_grip = a[6]
            obs, _, done, _ = env.step(a.tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
        if closure is None:
            cls = "TIMING"  # never closed
        elif closure["xy"] > args.xy_tol:
            cls = "XY_OFFSET"
        elif closure["dz"] > 0.04:
            cls = "TOO_HIGH"
        elif closure["dz"] < -0.02:
            cls = "TOO_LOW"
        elif lifted > 0.10:
            cls = "FIRM"
        else:
            cls = "TIMING"
        classes.append(cls)
        cs = closure or {"xy": -1, "dz": 0, "gerr": -1, "eeg": -1}
        env.close()
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={nm(T):13s} | bind_err={cs['gerr']*100:4.0f}cm "
              f"motor_err={cs['eeg']*100:4.0f}cm closure_xy={cs['xy']*100:4.0f}cm lift={lifted*100:4.0f}cm -> {cls}", flush=True)
    from collections import Counter
    c = Counter(classes); n = len(classes)
    print(f"\n=== GRASP FAILURE BREAKDOWN (N={n}, seed={args.seed}) ===", flush=True)
    for k in ["FIRM", "XY_OFFSET", "TOO_HIGH", "TOO_LOW", "TIMING"]:
        if c.get(k): print(f"  {k:11s} {c[k]:2d}/{n} ({c[k]/n*100:.0f}%)", flush=True)
    print("DIAG_GRASP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
