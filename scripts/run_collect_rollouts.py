"""Collect Pi0.5 rollouts on LIBERO with VLM hidden state extraction.

Uses create_trained_policy to load the model with proper input/output
transforms (normalization, tokenization, action unnormalization).
Extracts VLM features from the policy's infer() output.

Usage:
    PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
    /workspace/openpi/.venv/bin/python -u scripts/run_collect_rollouts.py \
        --task-suite libero_90 --num-trials 5 --output-dir data/contact_mpc/rollouts
"""

import argparse
import collections
import logging
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

# Allow torch.load to unpickle numpy arrays in LIBERO init state files
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

from openpi.policies import policy_config
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, force=True)

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
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    parser.add_argument("--task-suite", default="libero_90")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output-dir", default="data/contact_mpc/rollouts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args()


def _quat2axisangle(quat):
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS.get(args.task_suite)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load policy with proper transforms (normalization, tokenization, etc.)
    print("Loading Pi0.5 policy with transforms...", flush=True)
    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
    )
    print("Policy loaded.", flush=True)

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = args.max_tasks or task_suite.n_tasks
    print(f"Task suite: {args.task_suite}, {num_tasks} tasks, {args.num_trials} trials each", flush=True)

    # Collect rollouts
    all_hidden_states = []
    all_action_chunks = []
    all_episode_ids = []
    all_task_ids = []
    all_timesteps = []
    all_is_success = []

    episode_counter = 0
    total_successes = 0
    total_episodes = 0

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

        for trial_idx in range(min(args.num_trials, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx])
            action_plan = collections.deque()

            episode_records = []
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                if not action_plan:
                    t_start = time.time()

                    # Build observation dict matching the eval script format
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)
                    )
                    state = np.concatenate((
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))

                    # Call policy.infer() — handles all transforms
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": str(task_description),
                    }
                    result = policy.infer(element)
                    action_chunk = result["actions"]  # [action_horizon, 7] — already unnormalized
                    h = result.get("vlm_features")  # [2048] or None

                    # Record
                    episode_records.append({
                        "hidden_state": h if h is not None else np.zeros(2048, dtype=np.float32),
                        "action_chunk": action_chunk[:args.replan_steps].astype(np.float32),
                        "sim_timestep": t,
                    })

                    action_plan.extend(action_chunk[:args.replan_steps])
                    n_decisions = len(episode_records)
                    print(f"    t={t} decision#{n_decisions}: {time.time()-t_start:.2f}s", flush=True)

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    total_successes += 1
                    break
                t += 1

            total_episodes += 1
            is_success = bool(done)

            for rec in episode_records:
                all_hidden_states.append(rec["hidden_state"])
                all_action_chunks.append(rec["action_chunk"])
                all_episode_ids.append(episode_counter)
                all_task_ids.append(task_id)
                all_timesteps.append(rec["sim_timestep"])
                all_is_success.append(is_success)

            episode_counter += 1

            if total_episodes % 5 == 0:
                print(f"  Episodes: {total_episodes}, Successes: {total_successes} "
                      f"({total_successes/total_episodes*100:.1f}%), "
                      f"Task {task_id+1}/{num_tasks}", flush=True)

        env.close()

    print(f"\nTotal: {total_episodes} episodes, {total_successes} successes "
          f"({total_successes/total_episodes*100:.1f}%)", flush=True)
    print(f"Total decision points: {len(all_hidden_states)}", flush=True)

    save_path = output_dir / f"rollouts_{args.task_suite}.npz"
    np.savez_compressed(
        save_path,
        hidden_states=np.stack(all_hidden_states),
        action_chunks=np.stack(all_action_chunks),
        episode_ids=np.array(all_episode_ids),
        task_ids=np.array(all_task_ids),
        timesteps=np.array(all_timesteps),
        is_success=np.array(all_is_success),
    )
    print(f"Saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
