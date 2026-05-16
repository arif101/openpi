"""Verify configured MPPI cost targets against actual baseline trajectories.

Loads a LIBERO task, runs Pi0.5 baseline for one episode, and logs:
  - The tracked body's xyz at every timestep
  - All non-robot body positions (so you can compare candidates)
  - The configured target from reason_v3_targets.yaml (mode + goal_xyz +
    track_bodies)

For mode=object tasks, the tracked body is whichever candidate is currently
farthest from goal_xyz at each step (matches MPPI's runtime selection). For
mode=ee, tracks the end-effector.

Verdict logic: if the tracked body got within 10cm of goal_xyz during the
trajectory, the target is reachable and relevant. Within 20cm = roughly
right. Otherwise probably wrong.

Usage:
    PYTHONPATH=src:third_party/libero uv run python3 -u \\
        scripts/verify_libero_targets.py \\
        --task-suite libero_90 --task-idx 6 --num-trials 1
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import sys
import time

import numpy as np
import torch
import yaml

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RESIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--task-suite", required=True)
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--perturbation-cm", type=float, default=0.0,
                   help="0 = no perturbation (clean baseline trajectory)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=520)
    p.add_argument("--targets-yaml", default="scripts/reason_v3_targets.yaml")
    return p.parse_args()


def quat2axisangle(quat):
    q = quat.copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def build_obs(obs, task_text):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE, RESIZE))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": str(task_text)}


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = bm.get_task_init_states(args.task_idx)

    # Load configured target (if any)
    configured_goal = None
    configured_mode = None
    configured_track_names: list[str] = []
    targets_path = pathlib.Path(args.targets_yaml)
    if targets_path.exists():
        targets = yaml.safe_load(targets_path.read_text()) or {}
        suite = targets.get(args.task_suite) or {}
        entry = suite.get(args.task_idx) or suite.get(str(args.task_idx))
        if entry is not None and "goal_xyz" in entry:
            configured_goal = np.array(entry["goal_xyz"], dtype=np.float64)
            configured_mode = entry.get("mode", "object")
            configured_track_names = entry.get("track_bodies") or []

    print(f"\n{'='*70}")
    print(f"Task: {task.language}")
    print(f"BDDL: {bddl.name}")
    print(f"{'='*70}")
    if configured_goal is not None:
        print(f"Configured: mode={configured_mode} goal_xyz={configured_goal}")
        if configured_track_names:
            print(f"            track_bodies={configured_track_names}")
    else:
        print(f"No configured target in {args.targets_yaml}")

    print("\nLoading Pi0.5...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint)
    print("Loaded.", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                              camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    sim = env.env.sim
    model = sim.model

    # Get initial body positions for context
    print("\n--- Non-robot bodies at init (reference) ---")
    robot_prefixes = ("robot0_", "gripper0_", "panda", "link", "world")
    object_init = {}
    env.reset()
    env.set_init_state(init_states[0])
    for b in range(1, model.nbody):
        name = model.body_id2name(b) or ""
        if name and not any(name.startswith(p) for p in robot_prefixes):
            object_init[name] = sim.data.body_xpos[b].copy()
    for name, xyz in sorted(object_init.items()):
        marker = ""
        if configured_goal is not None:
            dist = np.linalg.norm(xyz - configured_goal)
            if dist < 0.05:
                marker = "  ← matches configured goal_xyz"
        print(f"  {name:35s} [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]{marker}")

    # Resolve tracked body ids (for mode=object) so we can log per-step pos
    tracked_ids: list[int] = []
    if configured_mode == "object" and configured_track_names:
        for nm in configured_track_names:
            try:
                tracked_ids.append(int(model.body_name2id(nm)))
            except Exception:
                print(f"  [warn] track body '{nm}' not found in model")
    ee_body_id = -1
    try:
        ee_body_id = int(model.body_name2id("robot0_eef"))
    except Exception:
        ee_body_id = model.nbody - 1

    perturb_m = args.perturbation_cm / 100.0
    perturb_rng = np.random.default_rng(args.seed + 1000)

    for trial_idx in range(min(args.num_trials, len(init_states))):
        print(f"\n{'-'*70}")
        print(f"Trial {trial_idx + 1}")
        print(f"{'-'*70}")

        env.reset()
        init = init_states[trial_idx]
        if perturb_m > 0:
            state = init.clone() if hasattr(init, "clone") else init.copy()
            for j in range(model.njnt):
                if model.jnt_type[j] == 0:  # FREE
                    a = model.jnt_qposadr[j]
                    state[a:a+3] += perturb_rng.normal(0, perturb_m, size=3)
            obs = env.set_init_state(state)
        else:
            obs = env.set_init_state(init)

        plan = collections.deque()
        done = False
        t = 0
        ee_history = []      # (t, xyz)
        tracked_history = [] # (t, body_name, xyz) — whichever body MPPI would track
        episode_start = time.time()

        while t < args.max_steps + 10:
            if t < 10:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if not plan:
                element = build_obs(obs, task.language)
                result = policy.infer(dict(element))
                action_chunk = np.asarray(result["actions"], dtype=np.float32)
                plan.extend(action_chunk[:5])

            action = plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            ee_xyz = sim.data.body_xpos[ee_body_id].copy()
            ee_history.append((t, ee_xyz))
            # Pick whichever tracked body is farthest from goal (matches MPPI logic).
            if configured_mode == "object" and tracked_ids and configured_goal is not None:
                best_id = tracked_ids[0]
                best_d = -1.0
                for bid in tracked_ids:
                    d = float(np.linalg.norm(sim.data.body_xpos[bid] - configured_goal))
                    if d > best_d:
                        best_d, best_id = d, bid
                tracked_history.append(
                    (t, model.body_id2name(best_id) or f"body_{best_id}",
                     sim.data.body_xpos[best_id].copy())
                )
            elif configured_mode == "ee":
                tracked_history.append((t, "robot0_eef", ee_xyz))
            if done:
                break
            t += 1

        wall = time.time() - episode_start
        outcome = "SUCCESS" if done else "FAIL"
        print(f"\nOutcome: {outcome} after {t} steps ({wall:.1f}s)")
        print(f"EE start:      [{ee_history[0][1][0]:+.3f}, {ee_history[0][1][1]:+.3f}, {ee_history[0][1][2]:+.3f}]")
        print(f"EE final:      [{ee_history[-1][1][0]:+.3f}, {ee_history[-1][1][1]:+.3f}, {ee_history[-1][1][2]:+.3f}]")

        # Sample every ~10% of trajectory
        n = len(ee_history)
        sample_pts = [int(n * f) for f in (0.25, 0.5, 0.75, 1.0)]
        print(f"\nEE trajectory samples:")
        for idx in sample_pts:
            if idx < n:
                tt, xyz = ee_history[idx - 1]
                print(f"  t={tt:4d}  [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]")

        if configured_goal is not None:
            # Always report EE distance (informative even in object mode).
            ee_d_min = min(np.linalg.norm(xyz - configured_goal) for _, xyz in ee_history)
            ee_d_final = float(np.linalg.norm(ee_history[-1][1] - configured_goal))
            print(f"\nEE → goal_xyz: final {ee_d_final:.3f}m, min {ee_d_min:.3f}m")

            # The MPPI-equivalent metric: tracked body → goal.
            if tracked_history:
                t_min = min(float(np.linalg.norm(xyz - configured_goal)) for _, _, xyz in tracked_history)
                t_final_t, t_final_name, t_final_xyz = tracked_history[-1]
                t_final = float(np.linalg.norm(t_final_xyz - configured_goal))
                # Stage transitions: report when the "currently farthest" body switches
                switches = sum(
                    1 for i in range(1, len(tracked_history))
                    if tracked_history[i][1] != tracked_history[i - 1][1]
                )
                print(f"tracked body → goal: final {t_final:.3f}m ({t_final_name}), "
                      f"min {t_min:.3f}m over trajectory; "
                      f"{switches} body switches")

                print(f"\nVerdict (tracked-body metric, what MPPI sees):")
                if t_min < 0.10:
                    print(f"  ✓ tracked body got within 10cm of goal — target is reachable + relevant")
                elif t_min < 0.20:
                    print(f"  ≈ within 20cm — target is roughly right, may want to refine")
                else:
                    print(f"  ✗ never within 20cm — goal_xyz or track_bodies probably wrong")
            else:
                print(f"\nVerdict (EE metric):")
                if ee_d_min < 0.10:
                    print(f"  ✓ EE got within 10cm of goal — target is reachable + relevant")
                elif ee_d_min < 0.20:
                    print(f"  ≈ EE within 20cm — target is roughly right")
                else:
                    print(f"  ✗ EE never within 20cm — target may be wrong")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
