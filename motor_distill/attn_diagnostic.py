"""Diagnostic: WHERE does pi0.5's action depend on the image, and does it follow
language? Occlusion-sensitivity heat map (causal 'where it looks') + the decisive
targeted test for the VISUAL-ATTRACTOR hypothesis.

For a scene with the demonstrated/attractor object (akita_black_bowl) AND a second
graspable object (wine_bottle), we run pi0.5 under THREE instructions:
  - original (demonstrated task, about the bowl)
  - counterfactual 'pick up the wine bottle'
  - counterfactual 'pick up the black bowl'
and measure, via OCCLUSION, how much each object drives the action:
  Delta_obj = || action(occlude obj) - action(full image) ||   (position part)
GROUNDING SCORE for an instruction naming object Y = Delta_Y / (Delta_bowl + Delta_bottle).

READ:
  - If under 'pick the wine bottle' the action still depends on the BOWL (Delta_bowl >>
    Delta_bottle, grounding score low) -> ATTENTIONAL/CAUSAL CAPTURE confirmed: the policy
    is driven by the memorized attractor, ignores language. -> a re-grounding fix that
    redirects WHAT-DRIVES-THE-ACTION is the right lever.
  - If the action DOES shift to the named object (grounding score high) but the grasp still
    fails -> the failure is downstream of where-it-looks -> re-grounding won't help.

Heat maps (full patch sweep) saved to data/keystone/attn/ for visual inspection.
No env / EGL needed — pure pi0.5 inference on corpus frames + Grounding-DINO for object boxes.
"""
from __future__ import annotations

import argparse
import glob
import math
import pathlib

import numpy as np


def _quat2axisangle(quat):
    quat = quat.copy(); quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_obs(base_img, wrist_img, ee_pos, ee_quat, gripper_qpos, prompt):
    from openpi_client import image_tools
    img = np.ascontiguousarray(base_img[::-1, ::-1])
    wr = np.ascontiguousarray(wrist_img[::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, 224, 224))
    state = np.concatenate((ee_pos, _quat2axisangle(np.asarray(ee_quat)), gripper_qpos))
    return {"observation/image": img, "observation/wrist_image": wr,
            "observation/state": state, "prompt": str(prompt)}


def act(policy, base_img, frame, prompt):
    obs = build_obs(base_img, frame["wrist_image"], frame["ee_pos"], frame["ee_quat"],
                    frame["gripper_qpos"], prompt)
    return np.asarray(policy.infer(obs)["actions"], np.float32)[:, :3]   # position part of chunk


def occlude(img, x0, y0, x1, y1, val=128):
    o = img.copy(); o[max(0, y0):y1, max(0, x0):x1] = val; return o


def detect_boxes(img, queries, device):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = "IDEA-Research/grounding-dino-tiny"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    pil = Image.fromarray(img)
    out_boxes = {}
    for name, q in queries.items():
        inp = proc(images=pil, text=q, return_tensors="pt").to(device)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_grounded_object_detection(
            o, inp.input_ids, box_threshold=0.2, text_threshold=0.2, target_sizes=[pil.size[::-1]])[0]
        if len(res["scores"]):
            b = res["boxes"][int(res["scores"].argmax())].cpu().numpy().astype(int)
            out_boxes[name] = b
    return out_boxes


def heatmap(policy, base_img, frame, prompt, a0, stride=20, patch=40):
    H, W = base_img.shape[:2]
    sal = np.zeros((H, W), np.float32)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            occ = occlude(base_img, x, y, x + patch, y + patch)
            ap = act(policy, occ, frame, prompt)
            d = float(np.linalg.norm(ap - a0))
            sal[y:y + patch, x:x + patch] = np.maximum(sal[y:y + patch, x:x + patch], d)
    return sal


def jet(s):
    """s in [0,1] -> jet RGB 0..255 (blue=low, green=mid, red=high)."""
    r = np.clip(1.5 - np.abs(4 * s - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * s - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * s - 1), 0, 1)
    return np.stack([r, g, b], -1) * 255.0


def save_overlay(base_img, sal, path):
    """Grad-CAM style: grayscale base (image stays legible) + jet heatmap, alpha by
    saliency so cold regions show the clear image and hot regions pop in color."""
    import imageio
    thr = np.percentile(sal, 95) + 1e-6                        # robust norm (ignore outliers)
    s = np.clip(sal / thr, 0, 1) ** 0.7                        # gamma -> mid-range contrast
    lum = base_img.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    gray = np.repeat(lum[..., None], 3, -1)
    heat = jet(s)
    a = np.clip(s, 0, 0.85)[..., None]                         # hot=color, cold=grayscale image
    out = (1 - a) * gray + a * heat
    imageio.imwrite(path, np.clip(out, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--data", default="data/keystone/pert0")
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--full-heatmap", action="store_true")
    p.add_argument("--heat-frames", type=int, default=3)
    args = p.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    out = pathlib.Path("data/keystone/attn"); out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(pathlib.Path(args.data) /
                   f"PHYS_OK_libero_10_task{args.task_idx}_*baseline*.npz")))[: args.n]
    QUERIES = {"bowl": "a bowl.", "bottle": "a wine bottle."}
    INSTR = {"original": None, "pick_bottle": "pick up the wine bottle",
             "pick_bowl": "pick up the black bowl"}
    TARGET = {"original": "bowl", "pick_bottle": "bottle", "pick_bowl": "bowl"}

    print(f"{'frame':6s} {'instruction':12s} {'d_bowl':>7s} {'d_bottle':>8s} {'grounding':>9s}", flush=True)
    agg = {k: [] for k in INSTR}
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        img = np.asarray(d["image"][0])                          # 256x256 agentview, first frame
        frame = {k: np.asarray(d[k][0]) for k in ("wrist_image", "ee_pos", "ee_quat", "gripper_qpos")}
        boxes = detect_boxes(img, QUERIES, dev)
        if "bowl" not in boxes or "bottle" not in boxes:
            print(f"  frame {fi}: missing box {list(boxes)} — skip", flush=True); continue
        for key, instr in INSTR.items():
            prompt = instr if instr is not None else str(d["prompt"])
            a0 = act(policy, img, frame, prompt)
            deltas = {}
            for name in ("bowl", "bottle"):
                x0, y0, x1, y1 = boxes[name]
                ap = act(policy, occlude(img, x0, y0, x1, y1), frame, prompt)
                deltas[name] = float(np.linalg.norm(ap - a0))
            tgt = TARGET[key]
            gscore = deltas[tgt] / (deltas["bowl"] + deltas["bottle"] + 1e-6)
            agg[key].append(gscore)
            print(f"{fi:6d} {key:12s} {deltas['bowl']:7.3f} {deltas['bottle']:8.3f} {gscore:9.2f}", flush=True)
            if args.full_heatmap and fi < args.heat_frames:
                sal = heatmap(policy, img, frame, prompt, a0)
                save_overlay(img, sal, out / f"heat_f{fi}_{key}.png")
    print("\n=== GROUNDING SCORE (delta on named object / total), mean over frames ===", flush=True)
    for k in INSTR:
        if agg[k]:
            print(f"  {k:12s}: {np.mean(agg[k]):.2f}  (high=follows language, ~0.5=ignores, low=anti)", flush=True)
    print("READ: if pick_bottle grounding << 0.5 -> action driven by the BOWL attractor, ignores language "
          "-> visual-attractor CAPTURE confirmed.", flush=True)
    print("ATTN_DIAG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
