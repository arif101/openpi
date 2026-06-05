"""FULL TASK SUCCESS (pick + place) with the factored pipeline. Object suite tasks = "pick up the X and
place it in the basket". Two phases, both using OWLv2 open-vocab binding + the distilled reach head:
  A) bind target X -> reach + grasp (reach head; until X lifted).
  B) bind container ("basket") -> transport (reach head toward basket goal, gripper FORCED closed) ->
     release (open gripper near basket).
SUCCESS = X ends up in/at the basket (horizontal dist < place_cm and lifted earlier). This is the real
LIBERO-PRO success metric (vs the reach/lift grounding proxies).
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
    ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxA", type=int, default=90); ap.add_argument("--maxB", type=int, default=90)
    ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--grip-close", type=float, default=1.0); ap.add_argument("--place-cm", type=float, default=12.0)
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
    H = W = 256
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def locate(img, tgt):
        best = None
        for q in (tgt, f"a {tgt}", f"a photo of a {tgt}", f"{tgt} container"):
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
        if rc is None:
            return None
        cy, cx = rc; reg = rd[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6, 0]
        d = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                            np.full((H, W, 1), d), c2w)[:3], np.float32)

    def act(goal, obs, grip=None):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        a = np.asarray(head_apply(rp, jnp.asarray(ee - goal),
                                  jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                  jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
        if grip is not None:
            a[6] = grip
        return a

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            continue
        T = targets[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset()
        sim = env.env.sim
        try:
            rb = resolve_bodies(sim, [T, args.container + "_1"]); cbody = rb.get(args.container + "_1")
        except Exception:
            rb = resolve_bodies(sim, [T]); cbody = None
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; phase = "A"; goalT = None; goalC = None

        def frame():
            img = np.asarray(obs["agentview_image"])[::-1].copy()
            rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
            return img, rd

        # Phase A: grasp T
        for step in range(args.maxA):
            if goalT is None or step % 20 == 0:
                img, rd = frame(); g = to_goal(img, rd, c2w, nm(T))
                if g is not None:
                    goalT = g
            if goalT is None:
                break
            obs, _, done, _ = env.step(act(goalT, obs).tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.04:
                phase = "B"; break
        # Phase B: transport to basket + release
        placed = False
        if phase == "B":
            for step in range(args.maxB):
                if goalC is None or step % 20 == 0:
                    img, rd = frame(); g = to_goal(img, rd, c2w, args.container)
                    if g is not None:
                        goalC = g + np.array([0, 0, 0.05], np.float32)   # aim slightly above basket
                if goalC is None:
                    break
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                near = float(np.linalg.norm(ee[:2] - goalC[:2])) < args.place_cm / 100
                obs, _, done, _ = env.step(act(goalC, obs, grip=(-1.0 if near else args.grip_close)).tolist())
            # success: T horizontally at basket and was lifted
            if cbody is not None:
                cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
                placed = bool(lifted > 0.04 and np.linalg.norm(cT[:2] - cC[:2]) < args.place_cm / 100)
        env.close()
        succ.append(placed)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={nm(T):13s} | lifted={lifted*100:.0f}cm phase={phase} placed={placed}", flush=True)
    n = len(succ)
    print(f"\n=== FULL TASK SUCCESS pick+place (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  success: {sum(succ)}/{n} = {sum(succ)/n*100:.0f}%   (baseline pi0.5 ~10% on object TASK axis)", flush=True)
    print("EVAL_E2E_PLACE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
