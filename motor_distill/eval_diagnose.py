"""Diagnose Pi0.5's OOD failure mode: is it the LOOKUP failure (ignores the moved
object, reaches the old spot) that the attractor fixes, or an execution/grasp
failure that re-targeting can't help?

Run live Pi0.5 on perturbed trials, track the gripper trajectory, and on each
FAILED trial classify by how close the gripper got to the ACTUAL (moved) object:
  LIFT            : success
  REACHED_NOGRASP : failed but gripper came within REACH_OK of the object
                    -> execution/grasp failure (attractor CANNOT help)
  MISSED_OBJECT   : failed, never reached the object
      STUCK       : ...and the gripper barely moved
      WENT_ELSEWHERE: ...and it moved but not toward the object (lookup-ish)

If most failures are REACHED_NOGRASP, the attractor was never applicable here and
we need a regime that induces the lookup failure. If most are MISSED_OBJECT /
WENT_ELSEWHERE, the lookup failure IS present and the attractor SHOULD help (so its
non-win is a precision problem, not a wrong-target problem).
"""
from __future__ import annotations

import argparse
import glob
import math
import pathlib

import numpy as np

import rekey
from eval_dmp_grasp import LIFT_M, GRASP_MAX_STEPS, derive

RESIZE = 224
REPLAN = 5
REACH_OK = 0.04        # gripper within 4cm of object center counts as "reached"
STUCK_DISP = 0.05      # max EE displacement < 5cm => barely moved


def _quat2axisangle(quat):
    quat = quat.copy(); quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_obs(obs, prompt):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wr = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(wr, RESIZE, RESIZE))
    state = np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wr,
            "observation/state": state, "prompt": str(prompt)}


def rollout(policy, env, init, tname, prompt):
    env.reset(); obs = env.set_init_state(init)
    sim = env.env.sim; bid = int(sim.model.body_name2id(tname))
    obj0 = sim.data.body_xpos[bid].astype(np.float64).copy()
    E0 = np.asarray(obs["robot0_eef_pos"], np.float64).copy()
    z0 = float(obj0[2])
    chunk = None; ci = 0
    min_d = 1e9; max_disp = 0.0; lifted = False
    for step in range(GRASP_MAX_STEPS):
        if chunk is None or ci >= REPLAN:
            chunk = np.asarray(policy.infer(build_obs(obs, prompt))["actions"], np.float32); ci = 0
        obs, _, done, _ = env.step(chunk[ci].tolist()); ci += 1
        E = np.asarray(obs["robot0_eef_pos"], np.float64)
        min_d = min(min_d, float(np.linalg.norm(E - obj0)))
        max_disp = max(max_disp, float(np.linalg.norm(E - E0)))
        if float(sim.data.body_xpos[bid][2]) - z0 > LIFT_M:
            lifted = True; break
    return lifted, min_d, max_disp


def classify(lifted, min_d, max_disp):
    if lifted:
        return "LIFT"
    if min_d < REACH_OK:
        return "REACHED_NOGRASP"
    if max_disp < STUCK_DISP:
        return "STUCK"
    return "WENT_ELSEWHERE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--ref-dir", default="data/keystone/pert0")
    p.add_argument("--perts", default="5,10")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    ref = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                 f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    _, _, tname, _, _ = derive(ref[:30], 16)
    prompt = str(np.load(ref[0], allow_pickle=True)["prompt"])

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)

    for pert in [int(x) for x in args.perts.split(",")]:
        test = sorted(glob.glob(str(pathlib.Path(f"data/keystone/pert{pert}") /
                      f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
        cats = {}; dists = []
        for f in test:
            init = np.load(f, allow_pickle=True)["init_state_libero"]
            lifted, min_d, max_disp = rollout(policy, env, init, tname, prompt)
            c = classify(lifted, min_d, max_disp); cats[c] = cats.get(c, 0) + 1
            if not lifted:
                dists.append(min_d)
            print(f"  pert{pert} {c:15s} min_d={min_d*100:.1f}cm disp={max_disp*100:.1f}cm", flush=True)
        n = len(test); nf = n - cats.get("LIFT", 0)
        print(f"\n=== pert{pert} Pi0.5 failure breakdown (N={n}) ===", flush=True)
        for c in ("LIFT", "REACHED_NOGRASP", "STUCK", "WENT_ELSEWHERE"):
            print(f"  {c:15s}: {cats.get(c,0)}", flush=True)
        if nf:
            md = np.array(dists)
            lookup = (md >= REACH_OK).sum()
            print(f"  of {nf} failures: {lookup} MISSED object (>={REACH_OK*100:.0f}cm, attractor-fixable), "
                  f"{nf-lookup} REACHED-but-failed (execution). median fail min_d={np.median(md)*100:.1f}cm", flush=True)
    print("DIAG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
