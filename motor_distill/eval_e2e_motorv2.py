"""Eval the learned motor v2 (relay single-goal + L1+BCE chunked) with TEMPORAL ENSEMBLING + foveation binder.
Relay latch: active goal = object until grasped (gripper closed, sustained), then container. Anti-memorization
preserved (object-agnostic goal-relative motor). Traced from the start to verify grasp/carry/place mechanics.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_motorv2.py --n 3 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_motor_v2 import motor_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--head", default="runs/motor_v2_head.pkl"); ap.add_argument("--container", default="basket")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hi", type=int, default=1024); ap.add_argument("--horizon", type=int, default=320)
    ap.add_argument("--thr", type=float, default=0.0); ap.add_argument("--place-cm", type=float, default=12.0)
    ap.add_argument("--ens-m", type=float, default=0.1); ap.add_argument("--trace", action="store_true")
    ap.add_argument("--obj-filter", default="")   # comma object tokens: only eval scenes whose TARGET noun matches (held-out-object split)
    ap.add_argument("--obj-center", action="store_true")   # place-precision: drive the grasped OBJECT to basket center (offset carry goal by grasp offset)
    args = ap.parse_args()
    import torch
    from transformers import CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    mh = jax.tree.map(jnp.asarray, pickle.load(open(args.head, "rb")))
    K = mh["w3"].shape[1] // 7
    H = args.hi; PLACE = args.place_cm / 100; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def propose(img, queries):
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
            if all(abs(cy-(k[1]+k[3])/2)+abs(cx-(k[0]+k[2])/2) > 0.13*H for k in kept): kept.append(b)
            if len(kept) >= 12: break
        return kept

    @torch.no_grad()
    def clip_pick(img, boxes, phrase):
        crops, cand = [], []
        for b in boxes:
            x0,y0,x1,y1 = b; m = 0.15*max(x1-x0, y1-y0)+8
            r0,c0 = max(0,int(y0-m)), max(0,int(x0-m)); r1,c1 = min(H,int(y1+m)), min(H,int(x1+m))
            if r1-r0 < 24 or c1-c0 < 24: continue
            crops.append(Image.fromarray(img[r0:r1, c0:c1]).resize((224,224))); cand.append((int((y0+y1)/2), int((x0+x1)/2), b))
        if not crops: return None
        ti = clipp(text=[f"a photo of {phrase}"], images=crops, return_tensors="pt", padding=True).to(dev)
        s = clip(**ti).logits_per_image.softmax(0)[:, 0].cpu().numpy()
        return cand[int(s.argmax())]

    def unproj(pick, rd, c2w):
        if pick is None: return None
        cy, cx, b = pick
        r0,c0,r1,c1 = max(0,int(b[1])), max(0,int(b[0])), min(H,int(b[3])), min(H,int(b[2]))
        reg = rd[r0:r1, c0:c1, 0]
        d = float(np.percentile(reg[reg>0], 15)) if (reg>0).any() else float(rd[cy,cx,0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy,cx],float), np.full((H,H,1),d), c2w)[:3], np.float32)

    motor_fn = jax.jit(motor_apply)
    DIN_HEAD = mh["w1"].shape[0]          # 9 = base, 12 = obj-state (object in-hand pose input)
    def predict(active, obs, obj_pos=None):
        ee = np.asarray(obs["robot0_eef_pos"], np.float32)
        q = jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)); g = jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))
        obj_rel = jnp.asarray((obj_pos - ee).astype(np.float32)) if (DIN_HEAD == 12 and obj_pos is not None) else None
        return np.asarray(motor_fn(mh, jnp.asarray(ee-active), q, g, obj_rel))  # (K,7)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    bind_ok, grasp_ok, full_ok, official_ok = [], [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; names = list(dict.fromkeys([T] + list(distractors)))
        if args.obj_filter and not any(t.strip() in re.sub(r"_\d+$", "", T) for t in args.obj_filter.split(",")):
            continue   # held-out-object split: skip scenes whose target isn't in the requested object set
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, names + [args.container + "_1"]); cbody = rb.get(args.container + "_1")
        except Exception:
            rb = resolve_bodies(sim, names); cbody = None
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H); c2w = np.linalg.inv(w2c)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H,H,1))[::-1].copy()
        qset = [nm(x) for x in names] + [args.container, "basket", "container", "bin"]
        boxes = propose(img, qset)
        goalT = unproj(clip_pick(img, boxes, nm(T)), rd, c2w)
        goalC = unproj(clip_pick(img, boxes, args.container), rd, c2w)
        objxyz = body_pos(sim, rb[T]).astype(np.float32)
        bind_ok.append(int(goalT is not None and np.linalg.norm(goalT[:2]-objxyz[:2]) < 0.05))
        if goalT is None or goalC is None:
            grasp_ok.append(0); full_ok.append(0); official_ok.append(0); env.close(); print(f"  {nm(T):14s} NO BIND", flush=True); continue
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; grasp_off = np.zeros(3, np.float32)
        recent = []                                          # (start_step, chunk) for temporal ensembling
        dt = args.trace and (len(grasp_ok) == 1)
        if dt: print(f"   [TRACE {nm(T)}] goalT={np.round(goalT,2)} goalC={np.round(goalC,2)}", flush=True)
        for step in range(args.horizon):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32); op = body_pos(sim, rb[T]).astype(np.float32)
            # PLACE-PRECISION: drive the OBJECT (not the EE) to basket center -> offset carry goal by grasp offset
            carry_goal = (goalC - grasp_off) if args.obj_center else goalC
            active = carry_goal if held else goalT
            recent.append((step, predict(active, obs, op))); recent = recent[-K:]   # op = object pose (privileged ceiling test for obj-state head)
            preds, ages = [], []
            for (s, ck) in recent:
                j = step - s
                if 0 <= j < K: preds.append(ck[j]); ages.append(j)
            wt = np.exp(-args.ens_m * np.array(ages)); wt /= wt.sum()
            a = (np.stack(preds) * wt[:, None]).sum(0)        # temporal-ensembled action
            grip = 1.0 if a[6] > 0 else -1.0
            close_cnt = close_cnt + 1 if grip > 0 else 0
            if (not held) and close_cnt > 8 and lifted > 0.02:
                held = True; grasp_off = (op - ee).astype(np.float32)   # object offset in gripper at grasp latch
            act = np.concatenate([a[:6], [grip]]).astype(np.float32)
            if dt and step % 15 == 0:
                print(f"      s{step:3d} held={int(held)} grip={grip:+.0f} ee2act={np.linalg.norm(ee-active)*100:3.0f}cm "
                      f"objH={(op[2]-z0)*100:4.0f}cm ee2obj={np.linalg.norm(ee[:2]-op[:2])*100:3.0f}cm", flush=True)
            obs, _, done, _ = env.step(act.tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if done: break
        placed = False; objdist = -1.0
        if cbody is not None:
            cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
            objdist = float(np.linalg.norm(cT[:2]-cC[:2])); placed = bool(lifted > 0.04 and objdist < PLACE)
        # OFFICIAL LIBERO/BDDL success (the metric the baselines are scored by) -- object IN the container
        try:
            off = bool(env.env._check_success())
        except Exception as e:
            off = None
            if len(bind_ok) == 1: print(f"   [official-success unavailable: {e}]", flush=True)
        official_ok.append(int(off) if off is not None else 0)
        grasp_ok.append(int(lifted > 0.04)); full_ok.append(int(placed)); env.close()
        print(f"  {nm(T):14s} bind={'Y' if bind_ok[-1] else '.'} grasp={'Y' if lifted>0.04 else '.'}({lifted*100:3.0f}cm) "
              f"held={int(held)} obj2basket={objdist*100:4.0f}cm proxy={'Y' if placed else '.'} OFFICIAL={'Y' if off else '.'}", flush=True)
    n = len(bind_ok)
    print(f"\n=== LEARNED MOTOR v2 (relay+L1/BCE+chunk+ensemble) + foveation binder (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  obj binding: {np.mean(bind_ok):.2f}   grasp: {np.mean(grasp_ok):.2f}   FULL(proxy): {np.mean(full_ok):.2f}   FULL(OFFICIAL BDDL): {np.mean(official_ok):.2f}", flush=True)
    print("EVAL_MOTORV2_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
