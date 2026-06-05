"""FULL pick+place with the UNIFIED phase-conditioned motor (one head) + OWLv2 binding.
Plan = [(target, phase0=grasp), (basket, phase1=place)]. The motor learned gripper timing from demos
(no scripted override). Success = named object ends in the basket. The scalable design test.
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
    ap.add_argument("--maxA", type=int, default=110); ap.add_argument("--maxB", type=int, default=110)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--place-cm", type=float, default=12.0)
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
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}", f"{tgt} container"):
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(owlm(**inp), threshold=args.thr,
                                                            target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]; best = (float(sc.max()), int((b[1]+b[3])/2), int((b[0]+b[2])/2))
        return None if best is None else (best[1], best[2])

    def to_goal(obs, sim, c2w, name, dz=0.0):
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
        rc = locate(img, name)
        if rc is None:
            return None
        cy, cx = rc; reg = rd[max(0, cy-6):cy+6, max(0, cx-6):cx+6, 0]
        d = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
        g = CU.transform_from_pixels_to_world(np.array([cy, cx], float), np.full((H, W, 1), d), c2w)[:3]
        return np.asarray(g, np.float32) + np.array([0, 0, dz], np.float32)

    def act(goal, obs, phase):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return np.array(head_apply(mp, jnp.asarray(ee - goal),
                                   jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                   jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32)),
                                   jnp.asarray([float(phase)])))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ, lifts, reaches = [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]; M = distractors[0] if distractors else None
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset()
        sim = env.env.sim
        try:
            bl = [T, args.container + "_1"] + ([M] if M else [])
            rb = resolve_bodies(sim, bl); cb = rb[args.container + "_1"]
        except Exception:
            env.close(); continue
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; gA = None; gB = None; rT, rM = 1e9, 1e9
        def upd_reach():
            nonlocal rT, rM
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            rT = min(rT, float(np.linalg.norm(E - body_pos(sim, rb[T]))))
            if M: rM = min(rM, float(np.linalg.norm(E - body_pos(sim, rb[M]))))
        for step in range(args.maxA):                                   # phase 0: grasp
            if gA is None or step % 20 == 0:
                g = to_goal(obs, sim, c2w, nm(T))
                if g is not None: gA = g
            if gA is None: break
            obs, _, done, _ = env.step(act(gA, obs, 0).tolist()); upd_reach()
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.04: break
        if lifted > 0.04:
            for step in range(args.maxB):                               # phase 1: transport+release
                if gB is None or step % 20 == 0:
                    g = to_goal(obs, sim, c2w, args.container, dz=0.06)
                    if g is not None: gB = g
                if gB is None: break
                obs, _, done, _ = env.step(act(gB, obs, 1).tolist())
        mpz = body_pos(sim, rb[T]); cpz = body_pos(sim, cb)
        placed = bool(lifted > 0.04 and np.linalg.norm(mpz[:2] - cpz[:2]) < args.place_cm / 100)
        reached = (rT < rM) if M else (rT < 0.08)
        env.close(); succ.append(placed); lifts.append(lifted > 0.04); reaches.append(reached)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={nm(T):13s} | reached={reached} lifted={lifted*100:.0f}cm placed={placed}", flush=True)
    n = len(succ)
    print(f"\n=== FULL pick+place, UNIFIED motor (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  NO-REGRESSION  reach: {sum(reaches)}/{n}={sum(reaches)/n*100:.0f}% (old 67%)  grasp/lift: {sum(lifts)}/{n}={sum(lifts)/n*100:.0f}% (old 47%)", flush=True)
    print(f"  NEW CAPABILITY success(place): {sum(succ)}/{n} = {sum(succ)/n*100:.0f}%", flush=True)
    print("EVAL_E2E_FULL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
