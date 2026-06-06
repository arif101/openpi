"""TRAINING-FREE gradient-steering arm (VLS-style, our foveation target) — the cheap first experiment that
de-risks the whole binding direction before any learned-injection training. Steer the FROZEN pi0.5 sampler toward
the foveation-grounded 3D target each denoising step (model.sample_actions_guided), with a RELAY active target
(object until grasped, then container). Score with the OFFICIAL LIBERO BDDL predicate. Sweep guide_w (lambda).
guide_w=0 MUST reproduce stock pi0.5 (~TASK baseline) = the sanity check. Beating CAG (21.7%) / matching VLS (+13%
LIBERO-PRO) with ZERO training would validate the binder end-to-end and set the baseline the learned KV-injection
must beat.

!!! UNTESTED — needs a GPU. Likely needs lambda tuning + possible action-frame fix (guidance applied in normalized
action space assuming LIBERO pos dims ~ world EE-delta direction). Run guide_w=0 first to confirm == baseline.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_guided.py --seed 7 --guide-w 0.1
"""
from __future__ import annotations
import argparse, glob, pathlib, re, functools
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--hi", type=int, default=1024); p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--guide-w", type=float, default=0.1)
    args = p.parse_args()
    import jax, jax.numpy as jnp, torch
    from transformers import CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download, nnx_utils
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)

    # --- monkeypatch the sampler to the guided one; per-step target dir read from a mutable cell ---
    GUIDE = {"dir": np.zeros(3, np.float32), "w": float(args.guide_w)}
    gsamp = nnx_utils.module_jit(policy._model.sample_actions_guided)
    def guided_sample(rng, observation, **kw):
        kw.pop("guide_dir", None); kw.pop("guide_w", None)
        return gsamp(rng, observation, guide_dir=jnp.asarray(GUIDE["dir"])[None, :], guide_w=GUIDE["w"], **kw)
    policy._sample_actions = guided_sample

    H = args.hi; nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def propose(img, queries):
        boxes = []
        for q in queries:
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(owlm(**inp), threshold=args.thr,
                                                            target_sizes=torch.tensor([[H, H]]).to(dev))[0]
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
            crops.append(Image.fromarray(img[r0:r1, c0:c1]).resize((224,224))); cand.append(b)
        if not crops: return None
        ti = clipp(text=[f"a photo of {phrase}"], images=crops, return_tensors="pt", padding=True).to(dev)
        s = clip(**ti).logits_per_image.softmax(0)[:, 0].cpu().numpy()
        return cand[int(s.argmax())]

    def unproj(b, rd, c2w):
        if b is None: return None
        cy, cx = int((b[1]+b[3])/2), int((b[0]+b[2])/2)
        r0,c0,r1,c1 = max(0,int(b[1])), max(0,int(b[0])), min(H,int(b[3])), min(H,int(b[2]))
        reg = rd[r0:r1, c0:c1, 0]; d = float(np.percentile(reg[reg>0], 15)) if (reg>0).any() else float(rd[cy,cx,0])
        return np.asarray(CU.transform_from_pixels_to_world(np.array([cy,cx],float), np.full((H,H,1),d), c2w)[:3], np.float32)

    def build_obs(base, wrist, ee, quat, grip, instr):
        from openpi_client import image_tools
        from cf_harness import _quat2axisangle
        im = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(base[::-1,::-1]), 224, 224))
        wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(wrist[::-1,::-1]), 224, 224))
        st = np.concatenate((ee, _quat2axisangle(np.asarray(quat)), grip))
        return {"observation/image": im, "observation/wrist_image": wr, "observation/state": st, "prompt": str(instr)}

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []
    for bf in bddls:
        instr_raw, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; names = list(dict.fromkeys([T] + list(distractors)))
        instr = f"pick up the {nm(T)} and place it in the {args.container}"
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
        try:
            rb = resolve_bodies(sim, names + [args.container + "_1"]); cb = rb[args.container + "_1"]
        except Exception:
            env.close(); continue
        w2c = CU.get_camera_transform_matrix(sim, "agentview", H, H); c2w = np.linalg.inv(w2c)
        img = np.asarray(obs["agentview_image"])[::-1].copy()
        rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H,H,1))[::-1].copy()
        qset = [nm(x) for x in names] + [args.container, "basket", "bin", "container"]
        boxes = propose(img, qset)
        goalT = unproj(clip_pick(img, boxes, nm(T)), rd, c2w); goalC = unproj(clip_pick(img, boxes, args.container), rd, c2w)
        if goalT is None: goalT = body_pos(sim, rb[T]).astype(np.float32)          # fallback so the arm still runs
        if goalC is None: goalC = body_pos(sim, cb).astype(np.float32)
        z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
        for step in range(args.horizon):
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            active = goalC if held else goalT
            d = active - ee; n = np.linalg.norm(d)
            GUIDE["dir"] = (d / n).astype(np.float32) if n > 1e-6 else np.zeros(3, np.float32)
            if chunk is None or ci >= args.replan:
                o_in = build_obs(np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"]),
                                 obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
            a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
            close_cnt = close_cnt + 1 if grip > 0 else 0
            if (not held) and close_cnt > 8 and lifted > 0.02: held = True
            obs, _, done, _ = env.step(a[:7].tolist())
            lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
            if done: break
        try: ok = bool(env.env._check_success())
        except Exception: ok = False
        succ.append(int(ok)); env.close()
        print(f"  {nm(T):14s} held={int(held)} liftT={lifted*100:3.0f}cm OFFICIAL={'Y' if ok else '.'}", flush=True)
    print(f"\n=== GUIDED pi0.5 (training-free, guide_w={args.guide_w}, N={len(succ)}, seed={args.seed}) place-T OFFICIAL: {np.mean(succ):.2f} ===", flush=True)
    print("PI05_GUIDED_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
