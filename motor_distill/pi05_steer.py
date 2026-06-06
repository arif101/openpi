"""HYBRID TEST: foveation binding -> STEER pi0.5's own motor to the counterfactual target T.
pi0.5's motor is near-perfect (100% w/ correct binding); the only TASK-axis failure is binding (it goes to its
memorized object). Steering = OBJECT-AGNOSTIC visual prompt: foveation locates T + basket, we MASK every other
object out of pi0.5's input image so it cannot bind to the memorized distractor -> it must go to T. Score with the
OFFICIAL bddl predicate (env._check_success checks T). Masking is by LOCATION (not identity) -> generalizes,
doesn't repeat memorization.
  --control: also run UNMASKED pi0.5 (the ~13% TASK-axis baseline) on the same scenes for contrast.

Run: PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/pi05_steer.py --seed 7 --control
"""
from __future__ import annotations
import argparse, glob, pathlib, re
import numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--container", default="basket"); p.add_argument("--n", type=int, default=10)
    p.add_argument("--hi", type=int, default=1024); p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--replan", type=int, default=5); p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--control", action="store_true")
    args = p.parse_args()
    import torch
    from transformers import CLIPModel, CLIPProcessor, Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    clipp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
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

    def center(b): return ((b[1]+b[3])/2, (b[0]+b[2])/2)

    def steer_mask(upr, T_box, keep_boxes, all_boxes):
        """Paint every proposed object box NOT in keep_boxes with the table color -> distractors removed from view."""
        out = upr.copy(); fill = np.median(upr.reshape(-1, 3), axis=0).astype(np.uint8)
        keptc = [center(b) for b in keep_boxes]
        for b in all_boxes:
            cy, cx = center(b)
            if any(abs(cy-kc[0]) + abs(cx-kc[1]) < 0.05*H for kc in keptc):
                continue                                       # this is T or basket -> keep
            x0,y0,x1,y1 = b; pad = int(0.10*max(x1-x0, y1-y0))
            r0,c0 = max(0,int(y0-pad)), max(0,int(x0-pad)); r1,c1 = min(H,int(y1+pad)), min(H,int(x1+pad))
            out[r0:r1, c0:c1] = fill                           # mask distractor by LOCATION (identity-free)
        return out

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    steer_ok, ctrl_ok = [], []
    for bf in bddls:
        instr_raw, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        T = targets[0]; names = list(dict.fromkeys([T] + list(distractors)))
        instr = f"pick up the {nm(T)} and place it in the {args.container}"

        def rollout(mask):
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=H)
            env.seed(args.seed); env.reset(); obs = env.reset(); sim = env.env.sim
            mask_img = None
            if mask:
                upr = np.asarray(obs["agentview_image"])[::-1, ::-1].copy()    # upright hi-res for detection
                qset = [nm(x) for x in names] + [args.container, "basket", "bin", "container"]
                boxes = propose(upr, qset)
                Tb = clip_pick(upr, boxes, nm(T)); Bb = clip_pick(upr, boxes, args.container)
                keep = [b for b in (Tb, Bb) if b is not None]
                mask_img = steer_mask(upr, Tb, keep, boxes) if keep else upr
            chunk = None; ci = 0
            for step in range(args.horizon):
                if chunk is None or ci >= args.replan:
                    base = np.asarray(obs["agentview_image"])
                    if mask:                                   # feed masked upright -> build_obs flips back to upright
                        cur = np.asarray(obs["agentview_image"])[::-1, ::-1].copy()
                        cur_keep = steer_mask(cur, None, keep, boxes) if mask_img is not None else cur
                        base = cur_keep[::-1, ::-1]
                    o_in = build_obs(base, np.asarray(obs["robot0_eye_in_hand_image"]),
                                     obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instr)
                    chunk = np.asarray(policy.infer(o_in)["actions"], np.float32); ci = 0
                obs, _, done, _ = env.step(chunk[ci][:7].tolist()); ci += 1
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            env.close(); return ok

        s_ok = rollout(mask=True); steer_ok.append(int(s_ok))
        line = f"  {nm(T):14s} STEER={'Y' if s_ok else '.'}"
        if args.control:
            c_ok = rollout(mask=False); ctrl_ok.append(int(c_ok)); line += f"  CONTROL(unmasked)={'Y' if c_ok else '.'}"
        print(line, flush=True)
    print(f"\n=== HYBRID foveation-STEER pi0.5 (N={len(steer_ok)}, seed={args.seed}) place-T OFFICIAL: {np.mean(steer_ok):.2f}"
          + (f"  | unmasked baseline: {np.mean(ctrl_ok):.2f}" if args.control else ""), flush=True)
    print("PI05_STEER_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
