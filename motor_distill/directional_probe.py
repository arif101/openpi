"""Sharper target-selection probe: does pi0.5's predicted MOTION point at the
bowl or the wine bottle under each instruction?

For each frame we take pi0.5's predicted action chunk, sum the world-frame position
deltas into an intended motion vector, project it onto the GROUND PLANE (x,y) to
remove the common 'reach down' component, and compare its direction to:
  EE -> bowl   and   EE -> bottle   (from the privileged object positions).
cos_bowl, cos_bottle => the policy 'targets' whichever object the motion aligns with.

READ:
  - 'pick the wine bottle' -> motion should align with the BOTTLE (cos_bottle > cos_bowl).
    If it STILL aligns with the bowl -> action driven by the memorized target, ignores
    language -> capture confirmed at the action level.
  - 'pick the black bowl' -> motion should align with the BOWL.
A clean language effect = the targeted object flips with the instruction. No flip = capture.
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


def cos2d(a, b):
    a = a[:2]; b = b[:2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-9))


def obj_index(names, *keys):
    for i, n in enumerate(names):
        if all(k in n for k in keys):
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--data", default="data/keystone/pert0")
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)

    files = sorted(glob.glob(str(pathlib.Path(args.data) /
                   f"PHYS_OK_libero_10_task{args.task_idx}_*baseline*.npz")))[: args.n]
    INSTR = {"original": None, "pick_bottle": "pick up the wine bottle",
             "pick_bowl": "pick up the black bowl"}
    print(f"{'frame':6s} {'instruction':12s} {'cos_bowl':>9s} {'cos_bottle':>10s} {'targets':>8s}", flush=True)
    flips = {"pick_bottle": [], "pick_bowl": []}
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        names = list(d["object_names"])
        bi = obj_index(names, "bowl"); ti = obj_index(names, "wine", "bottle")
        if bi is None or ti is None:
            print(f"  frame {fi}: missing obj idx — skip", flush=True); continue
        E = np.asarray(d["ee_pos"][0], np.float64)
        bowl = np.asarray(d["object_pos"][0, bi], np.float64)
        bottle = np.asarray(d["object_pos"][0, ti], np.float64)
        wrist = np.asarray(d["wrist_image"][0]); base = np.asarray(d["image"][0])
        for key, instr in INSTR.items():
            prompt = instr if instr is not None else str(d["prompt"])
            obs = build_obs(base, wrist, d["ee_pos"][0], d["ee_quat"][0], d["gripper_qpos"][0], prompt)
            a = np.asarray(policy.infer(obs)["actions"], np.float64)
            motion = a[:, :3].sum(0)                       # intended world displacement
            cb = cos2d(motion, bowl - E); cw = cos2d(motion, bottle - E)
            targets = "bottle" if cw > cb else "bowl"
            if key in flips:
                flips[key].append(targets)
            print(f"{fi:6d} {key:12s} {cb:9.2f} {cw:10.2f} {targets:>8s}", flush=True)
    print("\n=== TARGET SELECTION (does motion flip with instruction?) ===", flush=True)
    for key in ("pick_bottle", "pick_bowl"):
        if flips[key]:
            want = "bottle" if key == "pick_bottle" else "bowl"
            correct = sum(t == want for t in flips[key])
            print(f"  {key:12s}: motion targets '{want}' in {correct}/{len(flips[key])} frames", flush=True)
    print("READ: clean grounding = pick_bottle->bottle high AND pick_bowl->bowl high (motion flips). "
          "Capture = motion stays on the bowl regardless of instruction.", flush=True)
    print("DIR_PROBE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
