"""Primitive-basis full pick+place: GRASP primitive = the firmer reach head (7-11cm lifts), PLACE primitive
= the phase-conditioned motor head. Selected by phase. Tests whether an un-diluted grasp primitive (vs the
unified head's 4-6cm) gives a firm-enough grasp to survive transport -> place. No wrist camera. The MoE/
primitive-basis design (each primitive distilled separately -> no interference)."""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply as reach_apply
from distill_motor import head_apply as motor_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach", default="runs/reach_head.pkl"); ap.add_argument("--motor", default="runs/motor_head.pkl")
    ap.add_argument("--container", default="basket"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--maxA", type=int, default=120)
    ap.add_argument("--maxB", type=int, default=110); ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--place-cm", type=float, default=12.0); ap.add_argument("--grasp-dz", type=float, default=-0.02)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach, "rb")))
    mp = jax.tree.map(jnp.asarray, pickle.load(open(args.motor, "rb")))
    H = W = 256; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

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

    def to_goal(obs, sim, c2w, name, dz=0.0):
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
        rc = locate(img, name)
        if rc is None: return None
        cy, cx = rc; reg = rd[max(0, cy-6):cy+6, max(0, cx-6):cx+6, 0]
        d = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float), np.full((H, W, 1), d), c2w)[:3], np.float32) + np.array([0,0,dz], np.float32)

    def grasp_act(goal, obs):           # firmer reach primitive (no phase)
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return np.array(reach_apply(rp, jnp.asarray(ee - goal), jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                    jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))

    def place_act(goal, obs):           # place primitive (motor head, phase 1)
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        return np.array(motor_apply(mp, jnp.asarray(ee - goal), jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                    jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32)), jnp.asarray([1.0])))

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    lifts, succ = [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, [T, args.container + "_1"]); cb = rb[args.container + "_1"]
        except Exception:
            env.close(); continue
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, W); c2w = np.linalg.inv(w2c)
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; gA = None; gB = None
        for step in range(args.maxA):
            if gA is None or step % 20 == 0:
                g = to_goal(obs, sim, c2w, nm(T), dz=args.grasp_dz)
                if g is not None: gA = g
            if gA is None: break
            obs, _, done, _ = env.step(grasp_act(gA, obs).tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.06: break
        if lifted > 0.04:
            for step in range(args.maxB):
                if gB is None or step % 20 == 0:
                    g = to_goal(obs, sim, c2w, args.container, dz=0.06)
                    if g is not None: gB = g
                if gB is None: break
                obs, _, done, _ = env.step(place_act(gB, obs).tolist())
        mpz = body_pos(sim, rb[T]); cpz = body_pos(sim, cb)
        placed = bool(lifted > 0.04 and np.linalg.norm(mpz[:2] - cpz[:2]) < args.place_cm / 100)
        env.close(); lifts.append(lifted > 0.04); succ.append(placed)
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={nm(T):13s} | lifted={lifted*100:.0f}cm placed={placed}", flush=True)
    n = len(succ)
    print(f"\n=== PRIMITIVE-BASIS pick+place (N={n}) grasp {sum(lifts)}/{n}={sum(lifts)/n*100:.0f}%  PLACE {sum(succ)}/{n}={sum(succ)/n*100:.0f}% ===", flush=True)
    print("EVAL_E2E_PRIM_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
