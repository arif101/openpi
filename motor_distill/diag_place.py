"""R1 de-risk: WHY is full pick+place 0%?  Instrument the existing factored place pipeline
(eval_e2e_place) and classify where each episode dies, to decide if place-0% is a MOTOR problem
(object dropped in transport), a BINDING problem (basket not localized), or a REPRESENTATION problem
(reaches basket but release misses → the AnyPlace relative-SE(3) place head is the right fix).

Failure classes (mutually exclusive, first that applies):
  NO_GRASP            phase A never lifted the object >4cm  (upstream grasp 47% ceiling)
  BASKET_NOT_BOUND    OWLv2 never localized the container in phase B
  DROPPED_IN_TRANSIT  object fell back below grasp height while gripper still commanded-closed
  NEVER_REACHED       EE never got within place_cm (horiz) of the basket goal
  RELEASED_MISSED     reached basket + released but object ended >place_cm from basket
  SUCCESS             object ended within place_cm of basket (and was lifted)

Run on box:  PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/diag_place.py --n 12 --seed 7
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
    ap.add_argument("--transport-head", default="")   # if set, use a dedicated transport primitive for phase B
    ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--maxA", type=int, default=90); ap.add_argument("--maxB", type=int, default=120)
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
    tp = jax.tree.map(jnp.asarray, pickle.load(open(args.transport_head, "rb"))) if args.transport_head else rp
    print(f"phase-B head: {'transport_head' if args.transport_head else 'reach_head (shared)'}", flush=True)
    H = W = 256
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")
    PLACE = args.place_cm / 100

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

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    classes = []
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
            obs, _, done, _ = env.step(act(rp, goalT, obs).tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if lifted > 0.04:
                phase = "B"; break

        cls = "NO_GRASP"; min_ee2basket = 9.9; dropped = False; released = False; basket_bound = False
        goalC_raw = None; basket_true = None; ee_end = None; min_ee2true = 9.9
        if phase == "B":
            basket_true = body_pos(sim, cbody).astype(np.float32) if cbody is not None else None
            cls = "BASKET_NOT_BOUND"
            for step in range(args.maxB):
                if goalC is None or step % 20 == 0:
                    img, rd = frame(); g = to_goal(img, rd, c2w, args.container)
                    if g is not None:
                        goalC_raw = g.copy(); goalC = g + np.array([0, 0, 0.05], np.float32); basket_bound = True
                if goalC is None:
                    break
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                d_xy = float(np.linalg.norm(ee[:2] - goalC[:2]))
                min_ee2basket = min(min_ee2basket, d_xy)
                if basket_true is not None:
                    min_ee2true = min(min_ee2true, float(np.linalg.norm(ee[:2] - basket_true[:2])))
                near = d_xy < PLACE
                grip = (-1.0 if near else args.grip_close)
                if near:
                    released = True
                obs, _, done, _ = env.step(act(tp, goalC, obs, grip=grip).tolist())
                ee_end = np.asarray(obs["robot0_eef_pos"], np.float32)
                obj_h = body_pos(sim, rb[T])[2] - z0
                if (not released) and obj_h < 0.02:    # object fell while still "held"
                    dropped = True
            # binding-vs-motor probe: localized basket goal vs TRUE basket pos
            if basket_true is not None and goalC_raw is not None:
                gerr = float(np.linalg.norm(goalC_raw[:2] - basket_true[:2]))
                print(f"      [probe] basket_goal={np.round(goalC_raw,2)} true={np.round(basket_true,2)} "
                      f"goal_err={gerr*100:.0f}cm | minEE2goal={min_ee2basket*100:.0f}cm "
                      f"minEE2true={min_ee2true*100:.0f}cm ee_end={np.round(ee_end,2) if ee_end is not None else None}", flush=True)
            # classify phase-B outcome
            if not basket_bound:
                cls = "BASKET_NOT_BOUND"
            elif dropped:
                cls = "DROPPED_IN_TRANSIT"
            elif min_ee2basket >= PLACE:
                cls = "NEVER_REACHED"
            else:
                cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody) if cbody is not None else cT
                ok = bool(lifted > 0.04 and np.linalg.norm(cT[:2] - cC[:2]) < PLACE)
                cls = "SUCCESS" if ok else "RELEASED_MISSED"
        env.close()
        classes.append(cls)
        print(f"  {pathlib.Path(bf).stem[:24]:26s} T={nm(T):13s} | lift={lifted*100:4.0f}cm "
              f"minEE2basket={min_ee2basket*100:4.0f}cm drop={int(dropped)} rel={int(released)} -> {cls}", flush=True)

    from collections import Counter
    c = Counter(classes); n = len(classes)
    print(f"\n=== PLACE FAILURE BREAKDOWN (N={n}, seed={args.seed}) ===", flush=True)
    for k in ["SUCCESS", "NO_GRASP", "BASKET_NOT_BOUND", "DROPPED_IN_TRANSIT", "NEVER_REACHED", "RELEASED_MISSED"]:
        if c.get(k):
            print(f"  {k:18s} {c[k]:2d}/{n}  ({c[k]/n*100:.0f}%)", flush=True)
    print("DIAG_PLACE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
