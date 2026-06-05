"""Full pick+place with the FACTORED MoE primitives (reach -> transport -> place-servo).
Each phase uses a dedicated object-agnostic goal-relative expert; phase switches are still hardcoded
here (lift>4cm A->B; horizontal-align B->C) -- those become the learned responsibility gate next.

  A reach:    reach_head, goal=object,    grip learned, until object lifted >4cm
  B transport: transport_head, goal=basket(+above), grip FORCED closed, until EE aligned over basket
  C place:    transport_head drives POSITION (goal=basket, z lowered), place_servo_head drives the GRIPPER
              (learned WHEN-to-open) -- release is emergent, not a hand-set distance threshold.
SUCCESS = object ends within place_cm of basket (and was lifted).

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_v2.py --n 12 --seed 7
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
    ap.add_argument("--transport-head", default="runs/transport_head.pkl")
    ap.add_argument("--place-head", default="runs/place_servo_head.pkl")
    ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxA", type=int, default=90); ap.add_argument("--maxB", type=int, default=90)
    ap.add_argument("--maxC", type=int, default=60); ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--cam", default="agentview"); ap.add_argument("--align-cm", type=float, default=6.0)
    ap.add_argument("--place-cm", type=float, default=12.0); ap.add_argument("--descend", type=float, default=0.06)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    L = lambda f: jax.tree.map(jnp.asarray, pickle.load(open(f, "rb")))
    rp, tp, pp = L(args.reach_head), L(args.transport_head), L(args.place_head)
    H = W = 256; PLACE = args.place_cm / 100; ALIGN = args.align_cm / 100
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

    def act(params, goal, obs, grip=None):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        a = np.array(head_apply(params, jnp.asarray(ee - goal),
                                jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
        if grip is not None:
            a[6] = grip
        return a

    def grip_of(params, goal, obs):
        return float(act(params, goal, obs)[6])

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
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; goalT = None; goalC = None; phase = "A"

        def frame():
            img = np.asarray(obs["agentview_image"])[::-1].copy()
            rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
            return img, rd

        for step in range(args.maxA):                        # A: reach+grasp
            if goalT is None or step % 20 == 0:
                img, rd = frame(); g = to_goal(img, rd, c2w, nm(T))
                if g is not None: goalT = g
            if goalT is None: break
            obs, _, done, _ = env.step(act(rp, goalT, obs).tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.04: phase = "B"; break

        opened = False
        if phase == "B":
            for step in range(args.maxB):                    # B: transport until aligned over basket
                if goalC is None or step % 20 == 0:
                    img, rd = frame(); g = to_goal(img, rd, c2w, args.container)
                    if g is not None: goalC = g + np.array([0, 0, 0.05], np.float32)
                if goalC is None: break
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                obs, _, done, _ = env.step(act(tp, goalC, obs, grip=1.0).tolist())
                if float(np.linalg.norm(ee[:2] - goalC[:2])) < ALIGN:
                    phase = "C"; break
        if phase == "C":
            goalP = goalC - np.array([0, 0, args.descend], np.float32)   # descend target
            for step in range(args.maxC):                    # C: transport drives position, place-servo the gripper
                gcmd = grip_of(pp, goalC, obs)               # learned WHEN-to-open
                obs, _, done, _ = env.step(act(tp, goalP, obs, grip=gcmd).tolist())
                if gcmd < 0: opened = True
                if opened and step > 5: break

        placed = False
        if cbody is not None:
            cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
            placed = bool(lifted > 0.04 and np.linalg.norm(cT[:2] - cC[:2]) < PLACE)
        env.close(); succ.append(placed)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={nm(T):13s} | lift={lifted*100:3.0f}cm phase={phase} "
              f"opened={int(opened)} placed={placed}", flush=True)
    n = len(succ)
    print(f"\n=== FULL pick+place SUCCESS (factored MoE primitives, N={n}, seed={args.seed}) ===", flush=True)
    print(f"  success: {sum(succ)}/{n} = {sum(succ)/n*100:.0f}%   (CAG bar = 21.7%, prior hardcoded = 0%)", flush=True)
    print("EVAL_E2E_V2_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
