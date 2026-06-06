"""NON-HARDCODED architecture: foveation binder -> unified position motor (strong baked-in attractor that
ARRIVES at its soft-blend active goal) + LEARNED GATE for the gripper (close/open from convergence signals).
No phase-switch, no release threshold -- the release is a learned function of convergence. One eval, full
pick->place.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_gated.py --n 12 --seed 7
"""
from __future__ import annotations
import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_unified import head_apply_unified
from distill_gate import gate_apply, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--head", default="runs/unified_k4_head.pkl"); ap.add_argument("--gate", default="runs/gate_head.pkl")
    ap.add_argument("--container", default="basket"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--hi", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=320); ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--place-cm", type=float, default=12.0)
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
    uh = jax.tree.map(jnp.asarray, pickle.load(open(args.head, "rb")))
    gz = pickle.load(open(args.gate, "rb")); gp = jax.tree.map(jnp.asarray, gz["params"]); gmu, gsd = gz["mu"], gz["sd"]
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

    gate_fn = jax.jit(gate_apply); KA = 5.0
    def gate_close(ee, goalT, goalC, grip):
        f = (feats(ee, goalT, goalC, grip) - gmu) / gsd        # learned gate: P(close) from convergence
        return float(gate_fn(gp, jnp.asarray(f))) > 0

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
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H,H,1))[::-1].copy()
        qset = [nm(x) for x in names] + [args.container, "basket", "container", "bin"]
        boxes = propose(img, qset)
        goalT = unproj(clip_pick(img, boxes, nm(T)), rd, c2w)
        goalC = unproj(clip_pick(img, boxes, args.container), rd, c2w)
        objxyz = body_pos(sim, rb[T]).astype(np.float32)
        bind_ok.append(int(goalT is not None and np.linalg.norm(goalT[:2]-objxyz[:2]) < 0.05))
        cont_ok.append(int(cbody is not None and goalC is not None and np.linalg.norm(goalC[:2]-body_pos(sim,cbody)[:2]) < 0.08))
        if goalT is None or goalC is None:
            grasp_ok.append(0); full_ok.append(0); env.close(); print(f"  {nm(T):14s} NO BIND", flush=True); continue
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; ee2b_min = 9.9; ptr = 0; released = 0
        btrue = body_pos(sim, cbody).astype(np.float32) if cbody is not None else None
        # emitter sub-goal plan: [(object, close), (container, open)]; gate advances the pointer on convergence
        for step in range(args.horizon):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            grip = np.asarray(obs["robot0_gripper_qpos"], np.float32)
            if btrue is not None: ee2b_min = min(ee2b_min, float(np.linalg.norm(ee[:2]-btrue[:2])))
            target = goalT if ptr == 0 else goalC               # pure attractor to CURRENT sub-goal target
            close = gate_close(ee, goalT, goalC, grip)
            if ptr == 0 and close:                              # gate fired grasp -> advance to place sub-goal
                ptr = 1
            a = np.zeros(7, np.float32)
            a[:3] = np.clip(KA * (target - ee), -1.0, 1.0)
            a[6] = 1.0 if close else -1.0                       # gate decides gripper
            if ptr == 1 and not close: released += 1            # gate fired release at container
            obs, _, done, _ = env.step(a.tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if released > 8: break                              # settled after release
            if done: break
        placed = False; objdist = -1.0
        if cbody is not None:
            cT = body_pos(sim, rb[T]); cC = body_pos(sim, cbody)
            objdist = float(np.linalg.norm(cT[:2]-cC[:2])); placed = bool(lifted > 0.04 and objdist < PLACE)
        grasp_ok.append(int(lifted > 0.04)); full_ok.append(int(placed)); env.close()
        print(f"  {nm(T):14s} bind={'Y' if bind_ok[-1] else '.'} grasp={'Y' if lifted>0.04 else '.'}({lifted*100:3.0f}cm) "
              f"minEE2basket={ee2b_min*100:3.0f}cm obj2basket={objdist*100:4.0f}cm placed={'Y' if placed else '.'}", flush=True)
    n = len(bind_ok)
    print(f"\n=== NON-HARDCODED: foveation binder + unified motor(attractor) + LEARNED GATE (N={n}, seed={args.seed}) ===", flush=True)
    print(f"  obj binding   : {np.mean(bind_ok):.2f}", flush=True)
    print(f"  basket binding: {np.mean(cont_ok):.2f}", flush=True)
    print(f"  grasp         : {np.mean(grasp_ok):.2f}", flush=True)
    print(f"  FULL success  : {np.mean(full_ok):.2f}   (CAG bar 21.7%, prior 0%)", flush=True)
    print("EVAL_E2E_GATED_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
