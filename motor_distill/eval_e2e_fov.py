"""Deployment foveated binder + full pick-place. Binding bottleneck fix: render @1024, propose candidate
object regions (OWLv2, high recall), CROP each at high res, CLIP-disambiguate the NAMED target, lift to 3D
via median-depth + geometry. Then the existing factored motor stack (reach -> transport -> place-servo,
nadir release). Measures BINDING accuracy (goal on named obj <5cm), GRASP (named obj lifted), FULL success.

The motor heads use goal-relative STATE (not the image), so only the binder needs the 1024 view; objects are
static pre-grasp so we localize once at episode start.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_fov.py --n 12 --seed 7
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
    ap.add_argument("--place-head", default="runs/place_full_head.pkl")
    ap.add_argument("--place-full", action="store_true")   # use unified post-grasp place head (vs scaffold)
    ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hi", type=int, default=1024); ap.add_argument("--win", type=float, default=0.13)
    ap.add_argument("--maxA", type=int, default=110); ap.add_argument("--maxB", type=int, default=90)
    ap.add_argument("--maxC", type=int, default=60); ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--align-cm", type=float, default=10.0); ap.add_argument("--descend", type=float, default=0.10)
    ap.add_argument("--place-cm", type=float, default=12.0)
    ap.add_argument("--conv-grasp", action="store_true")   # close gripper on CONVERGENCE at object, not reach-head timing
    args = ap.parse_args()
    import torch
    from transformers import (CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection)
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    L = lambda f: jax.tree.map(jnp.asarray, pickle.load(open(f, "rb")))
    rp, tp = L(args.reach_head), L(args.transport_head)
    pf = L(args.place_head) if args.place_full else None
    H = args.hi; win = int(args.win * H); PLACE = args.place_cm / 100; ALIGN = args.align_cm / 100
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def propose(img, queries):
        """High-recall candidate boxes over all queries (OWLv2, low thr), NMS-deduped."""
        boxes = []
        for q in queries:
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, H]]).to(dev))[0]
            for sc, b in zip(r["scores"].cpu().numpy(), r["boxes"].cpu().numpy()):
                boxes.append((float(sc), b))
        boxes.sort(key=lambda x: -x[0]); kept = []
        for sc, b in boxes:
            cy, cx = (b[1]+b[3])/2, (b[0]+b[2])/2
            if all(abs(cy-(k[1]+k[3])/2) + abs(cx-(k[0]+k[2])/2) > win for k in kept):
                kept.append(b)
            if len(kept) >= 12: break
        return kept

    @torch.no_grad()
    def clip_pick(img, boxes, phrase):
        """Crop each candidate to ITS OWN box extent (adaptive: small->tight zoom, large->full extent),
        CLIP-sim to the named phrase, return (center_pixel, box) of best. General for any object size."""
        crops, cand = [], []
        for b in boxes:
            x0, y0, x1, y1 = b; m = 0.15 * max(x1-x0, y1-y0) + 8
            r0, c0 = max(0, int(y0-m)), max(0, int(x0-m)); r1, c1 = min(H, int(y1+m)), min(H, int(x1+m))
            if r1-r0 < 24 or c1-c0 < 24: continue
            crops.append(Image.fromarray(img[r0:r1, c0:c1]).resize((224, 224)))
            cand.append((int((y0+y1)/2), int((x0+x1)/2), b))
        if not crops: return None
        ti = clipp(text=[f"a photo of {phrase}"], images=crops, return_tensors="pt", padding=True).to(dev)
        s = clip(**ti).logits_per_image.softmax(0)[:, 0].cpu().numpy()
        return cand[int(s.argmax())]

    def unproj_px(pick, rd, c2w):
        """Robust near-surface depth: low percentile of depths over the WHOLE chosen box (nearest solid
        surface — grocery front face OR container near-rim) — general, robust to hollow/edge geometry."""
        if pick is None: return None
        cy, cx, b = pick
        r0, c0, r1, c1 = max(0, int(b[1])), max(0, int(b[0])), min(H, int(b[3])), min(H, int(b[2]))
        reg = rd[r0:r1, c0:c1, 0]
        d = float(np.percentile(reg[reg > 0], 15)) if (reg > 0).any() else float(rd[cy, cx, 0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                            np.full((H, H, 1), d), c2w)[:3], np.float32)

    def bind_all(img, rd, c2w, target_phrase, container_phrase, queries):
        """ONE general binder: propose over all instruction nouns (object + container), adaptive-crop each,
        CLIP-disambiguate the target AND the container from the SAME box set. Container is just another noun."""
        boxes = propose(img, queries)
        gT = unproj_px(clip_pick(img, boxes, target_phrase), rd, c2w)
        gC = unproj_px(clip_pick(img, boxes, container_phrase), rd, c2w)
        return gT, gC

    def act(params, goal, obs, grip=None):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        a = np.array(head_apply(params, jnp.asarray(ee - goal),
                                jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
        if grip is not None: a[6] = grip
        return a

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    bind_ok, cont_ok, grasp_ok, full_ok = [], [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; names = list(dict.fromkeys([T] + list(distractors)))
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, names + [args.container + "_1"]); cbody = rb.get(args.container + "_1")
        except Exception:
            rb = resolve_bodies(sim, names); cbody = None
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H); c2w = np.linalg.inv(w2c)

        def frame():
            img = np.asarray(obs["agentview_image"])[::-1].copy()
            rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, H, 1))[::-1].copy()
            return img, rd

        img, rd = frame()
        # proposals over all instruction nouns (grocery candidates + the container noun) — open-vocab, general
        qset = [nm(x) for x in names] + [args.container, "basket", "container", "bin"]
        goalT, goalC = bind_all(img, rd, c2w, nm(T), args.container, qset)
        # binding accuracy: goalT near true object, goalC near true basket?
        objxyz = body_pos(sim, rb[T]).astype(np.float32)
        bok = goalT is not None and float(np.linalg.norm(goalT[:2] - objxyz[:2])) < 0.05
        bind_ok.append(int(bok))
        ck = (cbody is not None and goalC is not None and
              float(np.linalg.norm(goalC[:2] - body_pos(sim, cbody)[:2])) < 0.08)
        cont_ok.append(int(ck))
        goalC_raw = goalC.copy() if goalC is not None else None      # basket pos (place head is goal-relative to it)
        if goalC is not None: goalC = goalC + np.array([0, 0, 0.05], np.float32)

        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; phase = "A"
        if goalT is not None and not args.conv_grasp:
            for step in range(args.maxA):
                obs, _, done, _ = env.step(act(rp, goalT, obs).tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if lifted > 0.04: phase = "B"; break
        elif goalT is not None:
            # convergence-gated grasp: approach with gripper OPEN until EE settles AT the object, then close + lift
            hist = []; closed = False; cstep = 0
            for step in range(args.maxA):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32); hist.append(ee.copy())
                d_xy = float(np.linalg.norm(ee[:2] - goalT[:2]))
                settled = len(hist) >= 6 and float(np.max(np.ptp(np.stack(hist[-6:]), axis=0))) < 0.015
                if (not closed) and d_xy < 0.05 and settled:
                    closed = True; cstep = step
                g = goalT if not closed else goalT + np.array([0, 0, 0.20], np.float32)   # lift after grasping
                obs, _, done, _ = env.step(act(rp, g, obs, grip=(1.0 if closed else -1.0)).tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if closed and step > cstep + 30: phase = "B"; break
        grasp_ok.append(int(lifted > 0.04))

        opened = False; obj_min_h = 9.9; ee2b_min = 9.9; reached = False
        btrue = body_pos(sim, cbody).astype(np.float32) if cbody is not None else None
        if args.place_full and phase == "B" and goalC_raw is not None:
            # ONE unified post-grasp head: carry -> descend -> release (gripper is the head's own output)
            opened_steps = 0
            for step in range(args.maxB + args.maxC):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                a = act(pf, goalC_raw, obs)                          # full action incl LEARNED gripper
                obs, _, done, _ = env.step(a.tolist())
                obj_min_h = min(obj_min_h, float(body_pos(sim, rb[T])[2] - z0))
                if btrue is not None: ee2b_min = min(ee2b_min, float(np.linalg.norm(ee[:2]-btrue[:2])))
                if ee2b_min < ALIGN: reached = True
                if a[6] < 0: opened = True; opened_steps += 1
                if opened and opened_steps > 25: break
            for _ in range(15):                                      # settle
                obs, _, done, _ = env.step(act(tp, goalC + np.array([0,0,0.18],np.float32), obs, grip=-1.0).tolist())
        elif phase == "B" and goalC is not None:
            for step in range(args.maxB):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                obs, _, done, _ = env.step(act(tp, goalC, obs, grip=1.0).tolist())
                obj_min_h = min(obj_min_h, float(body_pos(sim, rb[T])[2] - z0))
                if btrue is not None: ee2b_min = min(ee2b_min, float(np.linalg.norm(ee[:2]-btrue[:2])))
                if float(np.linalg.norm(ee[:2] - goalC[:2])) < ALIGN: phase = "C"; reached = True; break
            if phase == "C":
                goalP = goalC - np.array([0, 0, args.descend], np.float32)
                start_z = float(np.asarray(obs["robot0_eef_pos"], np.float32)[2]); z_min = np.inf; desc = False
                for step in range(args.maxC):
                    z = float(np.asarray(obs["robot0_eef_pos"], np.float32)[2]); z_min = min(z_min, z)
                    desc = desc or (start_z - z_min > args.descend - 0.02)
                    nadir = desc and z > z_min + 0.015
                    g_use = (goalC + np.array([0,0,0.12], np.float32)) if opened else goalP
                    obs, _, done, _ = env.step(act(tp, g_use, obs, grip=-1.0 if (nadir or opened) else 1.0).tolist())
                    obj_min_h = min(obj_min_h, float(body_pos(sim, rb[T])[2] - z0))
                    if nadir or opened: opened = True
                for _ in range(20):
                    obs, _, done, _ = env.step(act(tp, goalC + np.array([0,0,0.15],np.float32), obs, grip=-1.0).tolist())
        placed = False; objdist = -1.0; objz = -1.0
        if cbody is not None:
            cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
            objdist = float(np.linalg.norm(cT[:2] - cC[:2])); objz = float(cT[2])
            placed = bool(lifted > 0.04 and objdist < PLACE)
        full_ok.append(int(placed)); env.close()
        # slip = object dropped (<2cm) during transport while gripper held closed
        slip = (phase in ("B","C")) and obj_min_h < 0.02
        print(f"  {nm(T):14s} bind={'Y' if bok else '.'} grasp={'Y' if lifted>0.04 else '.'}({lifted*100:3.0f}cm) "
              f"reachBasket={'Y' if reached else '.'} slipTransit={'Y' if slip else '.'} "
              f"obj2basket={objdist*100:4.0f}cm placed={'Y' if placed else '.'}", flush=True)
    n = len(bind_ok)
    print(f"\n=== FOVEATED BINDER + full pick-place (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  obj binding <5cm    : {np.mean(bind_ok):.2f}   (OWLv2@256 was ~0.40)", flush=True)
    print(f"  basket binding <8cm : {np.mean(cont_ok):.2f}", flush=True)
    print(f"  grasp               : {np.mean(grasp_ok):.2f}", flush=True)
    print(f"  FULL success : {np.mean(full_ok):.2f}   (CAG bar 21.7%, prior 0%)", flush=True)
    print("EVAL_E2E_FOV_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
