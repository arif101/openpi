"""Robust counterfactual-diagnostic harness on REAL LIBERO-PRO TASK-axis scenes.

LIBERO-PRO TASK perturbation: same scene the policy memorized, but the BDDL
(:language ...) names a DIFFERENT (swapped) target and the (:goal ...) success
check is updated to match. So: instruction says "pan", policy memorized "moka pot",
scene contains BOTH. Following language -> success; memorized -> failure.

This module is the reusable foundation for our diagnostic suite:
  load_scene(bddl)         -> env, perturbed instruction, target objects, distractors
  run_episode(...)         -> success (env goal check) + which object was grasped
  box_oracle: mask all-but-target image (Grounding-DINO) -> "perfect cue" upper bound
Pluggable diagnostics (heatmap, directional) consume the same loader.

KEYSTONE QUESTION (box-oracle / D2): does a frozen pi0.5, given a PERFECT visual
cue (only the named object visible), grasp the RIGHT object on hard counterfactuals?
 -> validates the frozen-VLA + grounding-field architecture AND gives the paper's
    headline upper bound.
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import pathlib

import numpy as np


# ----------------------------- BDDL parsing -----------------------------
def parse_bddl(path):
    txt = pathlib.Path(path).read_text()
    lang = re.search(r"\(:language\s+(.+?)\)", txt, re.S)
    instruction = " ".join(lang.group(1).split()) if lang else ""
    objs = re.search(r"\(:objects\s+(.+?)\)\s*\(:", txt, re.S)
    object_bodies = re.findall(r"(\w+_\d+)\s+-\s+\w+", objs.group(1)) if objs else []
    ooi = re.search(r"\(:obj_of_interest\s+(.+?)\)", txt, re.S)
    interest = re.findall(r"(\w+_\d+)", ooi.group(1)) if ooi else []
    # graspable targets = objects named in obj_of_interest (exclude fixtures/regions)
    targets = [o for o in interest if o in object_bodies]
    distractors = [o for o in object_bodies if o not in targets]
    return instruction, object_bodies, targets, distractors


# ----------------------------- obs / policy -----------------------------
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


# ----------------------------- box-oracle mask -----------------------------
_GDINO = {}
def detect_box(img, query, device):
    import torch
    from PIL import Image
    if "m" not in _GDINO:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        mid = "IDEA-Research/grounding-dino-tiny"
        _GDINO["p"] = AutoProcessor.from_pretrained(mid)
        _GDINO["m"] = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    proc, model = _GDINO["p"], _GDINO["m"]
    pil = Image.fromarray(img)
    inp = proc(images=pil, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model(**inp)
    res = proc.post_process_grounded_object_detection(
        o, inp.input_ids, box_threshold=0.2, text_threshold=0.2, target_sizes=[pil.size[::-1]])[0]
    if len(res["scores"]) == 0:
        return None
    return res["boxes"][int(res["scores"].argmax())].cpu().numpy().astype(int)


def mask_except(img, box, pad=12, val=128):
    if box is None:
        return img
    H, W = img.shape[:2]
    out = np.full_like(img, val)
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    return out


# ----------------------------- episode -----------------------------
def body_z(sim, name):
    try:
        return float(sim.data.body_xpos[int(sim.model.body_name2id(name))][2])
    except Exception:
        return np.nan


def run_episode(policy, env, instruction, targets, distractors, init=None,
                horizon=300, replan=5, oracle_query=None, device="cuda"):
    env.reset()
    obs = env.set_init_state(init) if init is not None else env.reset()
    sim = env.env.sim
    bodies = targets + distractors
    z0 = {b: body_z(sim, b) for b in bodies}
    chunk = None; ci = 0; success = False
    lifted = {b: 0.0 for b in bodies}
    for step in range(horizon):
        if chunk is None or ci >= replan:
            base = np.asarray(obs["agentview_image"])
            if oracle_query is not None:
                box = detect_box(base, oracle_query, device)
                base = mask_except(base, box)
            o = build_obs(base, np.asarray(obs["robot0_eye_in_hand_image"]),
                          obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instruction)
            chunk = np.asarray(policy.infer(o)["actions"], np.float32); ci = 0
        obs, _, done, info = env.step(chunk[ci].tolist()); ci += 1
        for b in bodies:
            lifted[b] = max(lifted[b], body_z(sim, b) - z0[b])
        if done or (isinstance(info, dict) and info.get("success")):
            success = True; break
    # which object did it engage most (largest lift)?
    grasped = max(bodies, key=lambda b: lifted[b]) if bodies else None
    grasped_is_target = grasped in targets
    return {"success": success, "grasped": grasped, "grasped_is_target": grasped_is_target,
            "lift": {b: round(lifted[b], 3) for b in bodies}}


# ----------------------------- main / aggregate -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_10_task")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--oracle", action="store_true", help="also run box-oracle (mask all-but-target)")
    args = p.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    print(f"{len(bddls)} TASK-axis scenes\n", flush=True)
    rows = []
    for bf in bddls:
        instruction, objs, targets, distractors = parse_bddl(bf)
        if not targets:
            print(f"  skip (no graspable target): {pathlib.Path(bf).name}", flush=True); continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7)
        tgt_q = "a " + targets[0].rsplit("_", 1)[0].replace("_", " ")     # query for GDINO from body name
        base = run_episode(policy, env, instruction, targets, distractors, horizon=args.horizon, device=dev)
        rec = {"scene": pathlib.Path(bf).stem[:40], "instr": instruction[:45],
               "target": targets[0], "base_succ": base["success"], "base_grasp_tgt": base["grasped_is_target"]}
        if args.oracle:
            orc = run_episode(policy, env, instruction, targets, distractors, horizon=args.horizon,
                              oracle_query=tgt_q, device=dev)
            rec["oracle_succ"] = orc["success"]; rec["oracle_grasp_tgt"] = orc["grasped_is_target"]
        env.close()
        rows.append(rec)
        print(f"  {rec['scene']:42s} instr='{rec['instr']}' tgt={rec['target']:22s} "
              f"base_succ={rec['base_succ']} grasp_tgt={rec['base_grasp_tgt']}"
              + (f" | oracle_succ={rec.get('oracle_succ')} grasp_tgt={rec.get('oracle_grasp_tgt')}" if args.oracle else ""),
              flush=True)
    n = len(rows)
    if n:
        bs = sum(r["base_succ"] for r in rows); bg = sum(r["base_grasp_tgt"] for r in rows)
        print(f"\n=== LIBERO-PRO TASK counterfactual (N={n}) ===", flush=True)
        print(f"  baseline: success {bs}/{n}={bs/n*100:.0f}%  grasped-named-target {bg}/{n}={bg/n*100:.0f}%", flush=True)
        if args.oracle:
            os_ = sum(r.get("oracle_succ", False) for r in rows); og = sum(r.get("oracle_grasp_tgt", False) for r in rows)
            print(f"  BOX-ORACLE: success {os_}/{n}={os_/n*100:.0f}%  grasped-named-target {og}/{n}={og/n*100:.0f}%", flush=True)
            print(f"  -> oracle gap (does a perfect cue re-target the frozen motor?): "
                  f"grasp-target {bg/n*100:.0f}% -> {og/n*100:.0f}%", flush=True)
    print("CF_HARNESS_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
