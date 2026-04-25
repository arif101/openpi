"""LIBERO-PRO position perturbation evaluation.

Tests Pi0.5 robustness by shifting object positions in the initial state
by a specified amount (default 5cm). This replicates the LIBERO-PRO
position perturbation experiment.

The perturbation works by modifying the MuJoCo state vector:
- Robot joints (first N_robot qpos entries) are left unchanged
- Free-joint objects have 7 qpos entries each (xyz + quaternion)
- We add Gaussian noise to the xyz positions of all non-robot bodies

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/run_libero_pro.py \
        --task-suite libero_10 --num-trials 5 --perturbation-cm 5
"""

import argparse
import collections
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--perturbation-cm", type=float, default=5.0, help="Position perturbation in cm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--episode-timeout", type=int, default=300)
    parser.add_argument("--output-dir", default="data/contact_mpc/libero_pro")
    return parser.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def perturb_object_positions(env, init_state, perturbation_m, rng):
    """Perturb object positions in the MuJoCo state vector.

    Identifies free-joint bodies (objects that can move freely) and adds
    Gaussian noise to their xyz positions. Robot joints are unchanged.

    Args:
        env: The LIBERO environment (needed for MuJoCo model info).
        init_state: Original flattened MuJoCo state vector.
        perturbation_m: Standard deviation of position noise in meters.
        rng: numpy RandomState.

    Returns:
        Perturbed state vector.
    """
    state = init_state.clone() if hasattr(init_state, 'clone') else init_state.copy()
    sim = env.env.sim
    model = sim.model

    # MuJoCo state = [qpos, qvel, ...]
    nq = model.nq  # number of qpos entries
    nv = model.nv  # number of qvel entries

    # Find free joints (objects that can be positioned freely)
    # Free joints have 7 qpos entries: x, y, z, qw, qx, qy, qz
    qpos_offset = 0
    for joint_idx in range(model.njnt):
        joint_type = model.jnt_type[joint_idx]
        joint_name = model.joint_id2name(joint_idx) if hasattr(model, 'joint_id2name') else f"joint_{joint_idx}"

        if joint_type == 0:  # mjJNT_FREE = 0
            # This is a free joint — likely an object
            # Skip if it's the robot base (unlikely in LIBERO, robot is fixed)
            body_id = model.jnt_bodyid[joint_idx]
            body_name = model.body_id2name(body_id) if hasattr(model, 'body_id2name') else f"body_{body_id}"

            # Perturb xyz position (first 3 of the 7 qpos entries)
            pos_start = model.jnt_qposadr[joint_idx]
            noise = rng.normal(0, perturbation_m, size=3)
            state[pos_start:pos_start + 3] += noise

        # Advance qpos offset based on joint type
        # (not needed since we use jnt_qposadr directly)

    return state


def run_evaluation(policy, task_suite, args, perturb=False):
    """Run evaluation with or without perturbation."""
    num_tasks = args.max_tasks or task_suite.n_tasks
    max_steps = MAX_STEPS[args.task_suite]
    perturbation_m = args.perturbation_cm / 100.0
    rng = np.random.RandomState(args.seed + (1000 if perturb else 0))

    total_episodes = 0
    total_successes = 0
    task_results = {}

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_description = task.language
        task_successes = 0

        for trial_idx in range(min(args.num_trials, len(initial_states))):
            env.reset()

            init_state = initial_states[trial_idx]
            if perturb:
                init_state = perturb_object_positions(env, init_state, perturbation_m, rng)

            obs = env.set_init_state(init_state)
            action_plan = collections.deque()
            done = False
            t = 0
            episode_start = time.time()

            while t < max_steps + args.num_steps_wait:
                if time.time() - episode_start > args.episode_timeout:
                    break

                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not action_plan:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    result = policy.infer(element)
                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    total_successes += 1
                    task_successes += 1
                    break
                t += 1

            total_episodes += 1

        task_results[task_id] = {
            "task": task_description,
            "successes": task_successes,
            "trials": min(args.num_trials, len(initial_states)),
            "rate": task_successes / min(args.num_trials, len(initial_states)),
        }
        env.close()

        rate = total_successes / total_episodes * 100
        mode = "PERTURBED" if perturb else "BASELINE"
        print(f"  [{mode}] Task {task_id+1}/{num_tasks}: "
              f"{task_successes}/{min(args.num_trials, len(initial_states))} "
              f"({task_results[task_id]['rate']:.0%}) | "
              f"Running: {total_successes}/{total_episodes} ({rate:.1f}%) | "
              f"{task_description[:60]}", flush=True)

    return {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes * 100,
        "task_results": task_results,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite not in MAX_STEPS:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load policy
    print("Loading Pi0.5 policy...", flush=True)
    from openpi.shared import download
    download.maybe_download(args.checkpoint + "/assets")
    download.maybe_download(args.checkpoint + "/params")
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    print("Policy loaded.", flush=True)

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    # Run baseline (no perturbation)
    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE (no perturbation) on {args.task_suite}", flush=True)
    print(f"{'='*60}", flush=True)
    baseline = run_evaluation(policy, task_suite, args, perturb=False)

    # Run perturbed
    print(f"\n{'='*60}", flush=True)
    print(f"PERTURBED (+/- {args.perturbation_cm}cm) on {args.task_suite}", flush=True)
    print(f"{'='*60}", flush=True)
    perturbed = run_evaluation(policy, task_suite, args, perturb=True)

    # Report
    print(f"\n{'='*60}", flush=True)
    print(f"LIBERO-PRO RESULTS ({args.task_suite}, perturbation={args.perturbation_cm}cm)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Baseline:  {baseline['success_rate']:.1f}% ({baseline['total_successes']}/{baseline['total_episodes']})", flush=True)
    print(f"Perturbed: {perturbed['success_rate']:.1f}% ({perturbed['total_successes']}/{perturbed['total_episodes']})", flush=True)
    print(f"Drop:      {baseline['success_rate'] - perturbed['success_rate']:.1f} percentage points", flush=True)

    print(f"\nPer-task breakdown:", flush=True)
    for tid in sorted(baseline["task_results"].keys()):
        b = baseline["task_results"][tid]
        p = perturbed["task_results"][tid]
        drop = b["rate"] - p["rate"]
        print(f"  Task {tid}: baseline={b['rate']:.0%} → perturbed={p['rate']:.0%} (drop={drop:+.0%}) | {b['task'][:60]}", flush=True)

    # Save
    np.savez(
        output_dir / f"libero_pro_{args.task_suite}_{args.perturbation_cm}cm.npz",
        baseline_rate=baseline["success_rate"],
        perturbed_rate=perturbed["success_rate"],
        perturbation_cm=args.perturbation_cm,
        task_suite=args.task_suite,
    )
    print(f"\nSaved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
