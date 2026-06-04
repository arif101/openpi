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


def mask_box(img, box, pad=6, val=128):
    """Gray out ONLY the box region (de-attractor: remove the memorized object,
    keep the rest of the scene). LIBERO-CF Table-II shows this raises CF success."""
    if box is None:
        return img
    H, W = img.shape[:2]
    out = img.copy()
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad); x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    out[y0:y1, x0:x1] = val
    return out


def clean_query(body):
    return "a " + re.sub(r"_\d+$", "", body).replace("_", " ")


# ----------------------------- episode -----------------------------
def resolve_bodies(sim, names):
    """Map bddl object names (e.g. 'moka_pot_1') -> actual mujoco body names
    (e.g. 'moka_pot_1_main') by exact/suffix/substring match."""
    allb = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
    out = {}
    for n in names:
        if n in allb:
            out[n] = n
        elif n + "_main" in allb:
            out[n] = n + "_main"
        else:
            cand = [b for b in allb if b and n in b]
            out[n] = cand[0] if cand else None
    return out


def body_pos(sim, bname):
    if bname is None:
        return np.full(3, np.nan)
    return sim.data.body_xpos[int(sim.model.body_name2id(bname))].astype(np.float64).copy()


def run_episode(policy, env, instruction, targets, distractors, init=None,
                horizon=300, replan=5, suppress_queries=None, device="cuda"):
    env.reset()
    obs = env.set_init_state(init) if init is not None else env.reset()
    sim = env.env.sim
    bodies = targets + distractors
    rb = resolve_bodies(sim, bodies)                                   # bddl name -> mujoco body name
    z0 = {b: body_pos(sim, rb[b])[2] for b in bodies}
    chunk = None; ci = 0; success = False
    lifted = {b: 0.0 for b in bodies}
    reach = {b: 1e9 for b in bodies}                                  # min EE->object dist (grounding)
    for step in range(horizon):
        if chunk is None or ci >= replan:
            base = np.asarray(obs["agentview_image"])
            if suppress_queries:                                       # de-attractor: gray ONLY memorized objects
                for q in suppress_queries:
                    base = mask_box(base, detect_box(base, q, device))
            o = build_obs(base, np.asarray(obs["robot0_eye_in_hand_image"]),
                          obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"], instruction)
            chunk = np.asarray(policy.infer(o)["actions"], np.float32); ci = 0
        obs, _, done, info = env.step(chunk[ci].tolist()); ci += 1
        E = np.asarray(obs["robot0_eef_pos"], np.float64)
        for b in bodies:
            pos = body_pos(sim, rb[b])
            lifted[b] = max(lifted[b], pos[2] - z0[b])
            reach[b] = min(reach[b], float(np.linalg.norm(E - pos)))
        if done or (isinstance(info, dict) and info.get("success")):
            success = True; break
    reach_tgt = min(reach[t] for t in targets)
    reach_dist = min([reach[d] for d in distractors], default=1e9)    # nearest distractor (incl memorized obj)
    lift_tgt = max(lifted[t] for t in targets)
    lift_dist = max([lifted[d] for d in distractors], default=0.0)
    return {"success": success,
            "reached_target": reach_tgt < reach_dist,                  # gripper went to named obj, not distractor
            "reach_tgt_cm": round(reach_tgt * 100, 1), "reach_dist_cm": round(reach_dist * 100, 1),
            "lifted_target": lift_tgt > 0.03 and lift_tgt > lift_dist,  # actually picked the named obj up
            "lift_tgt_cm": round(lift_tgt * 100, 1), "lift_dist_cm": round(lift_dist * 100, 1)}


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
        base = run_episode(policy, env, instruction, targets, distractors, horizon=args.horizon, device=dev)
        rec = {"scene": pathlib.Path(bf).stem[:38], "instr": instruction[:42], "target": targets[0],
               "b_succ": base["success"], "b_reach": base["reached_target"], "b_lift": base["lifted_target"],
               "b_rt": base["reach_tgt_cm"], "b_rd": base["reach_dist_cm"]}
        if args.oracle:
            orc = run_episode(policy, env, instruction, targets, distractors, horizon=args.horizon,
                              suppress_queries=[clean_query(d) for d in distractors], device=dev)
            rec["o_reach"] = orc["reached_target"]; rec["o_lift"] = orc["lifted_target"]; rec["o_succ"] = orc["success"]
        env.close()
        rows.append(rec)
        print(f"  {rec['scene']:40s} tgt={rec['target']:20s} | reach_tgt={rec['b_rt']}cm reach_distr={rec['b_rd']}cm "
              f"reached_tgt={rec['b_reach']} lifted_tgt={rec['b_lift']} succ={rec['b_succ']}"
              + (f" || ORACLE reached={rec.get('o_reach')} lifted={rec.get('o_lift')} succ={rec.get('o_succ')}" if args.oracle else ""),
              flush=True)
    n = len(rows)
    if n:
        f = lambda k: sum(bool(r.get(k)) for r in rows)
        print(f"\n=== LIBERO-PRO TASK counterfactual (N={n}) — grounding metrics ===", flush=True)
        print(f"  BASELINE: reached-named-target {f('b_reach')}/{n}={f('b_reach')/n*100:.0f}%  "
              f"lifted-named-target {f('b_lift')}/{n}={f('b_lift')/n*100:.0f}%  success {f('b_succ')}/{n}={f('b_succ')/n*100:.0f}%", flush=True)
        if args.oracle:
            print(f"  BOX-ORACLE: reached {f('o_reach')}/{n}={f('o_reach')/n*100:.0f}%  "
                  f"lifted {f('o_lift')}/{n}={f('o_lift')/n*100:.0f}%  success {f('o_succ')}/{n}={f('o_succ')/n*100:.0f}%", flush=True)
            print(f"  -> KEYSTONE: does a perfect cue re-target the frozen motor? "
                  f"reached-target {f('b_reach')/n*100:.0f}% -> {f('o_reach')/n*100:.0f}%, "
                  f"lifted {f('b_lift')/n*100:.0f}% -> {f('o_lift')/n*100:.0f}%", flush=True)
    print("CF_HARNESS_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
