"""Collect Pi0.5 rollouts on LIBERO-90 with VLM hidden state extraction.

Runs frozen Pi0.5 in the LIBERO simulator, records (hidden_state, action,
success/failure) at every decision point. Output is used for:
- Training the pairwise value function (success vs failure pairs)
- Training F_pi0 (world model on Pi0.5's action distribution)
- Re-evaluating KS3 with real success/failure signal

Usage:
    PYTHONPATH=/workspace/openpi/src /workspace/openpi/.venv/bin/python \
        scripts/run_collect_rollouts.py \
        --task-suite libero_90 \
        --num-trials 5 \
        --output-dir data/contact_mpc/rollouts
"""

import argparse
import collections
import logging
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import torch
from openpi_client import image_tools

# Allow torch.load to unpickle numpy arrays in LIBERO init state files
# (PyTorch 2.7 defaults to weights_only=True which rejects numpy globals)
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, "weights_only": False})

# These imports require LIBERO to be installed
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.contact_mpc.features.extractor import extract_features_from_dict
from openpi.contact_mpc.features.dataset import FeatureDataset, detect_contact_timesteps
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
from openpi.shared import nnx_utils

# Configure logging after all imports
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    parser.add_argument("--task-suite", default="libero_90")
    parser.add_argument("--num-trials", type=int, default=5, help="Trials per task")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output-dir", default="data/contact_mpc/rollouts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for debugging)")
    return parser.parse_args()


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_pi05_model(checkpoint_path: str):
    """Load frozen Pi0.5 and return model + jitted sample_actions."""
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    params_path = download.maybe_download(checkpoint_path)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params)
    model.eval()

    # JIT compile sample_actions for speed
    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)

    return model, sample_actions_jit, config


def build_observation_dict(obs, task_description):
    """Convert LIBERO observation to model input dict."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))

    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))

    return {
        "image": {
            "base_0_rgb": img[np.newaxis],
            "left_wrist_0_rgb": wrist_img[np.newaxis],
            "right_wrist_0_rgb": np.zeros_like(img[np.newaxis]),
        },
        "image_mask": {
            "base_0_rgb": np.array([True]),
            "left_wrist_0_rgb": np.array([True]),
            "right_wrist_0_rgb": np.array([True]),
        },
        "state": np.pad(state, (0, 32 - len(state)))[np.newaxis].astype(np.float32),
    }, state


def main():
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS.get(args.task_suite)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    # Load model
    print(f"Loading Pi0.5 model...", flush=True)
    model, sample_actions_jit, config = load_pi05_model(args.checkpoint)
    rng = jax.random.key(0)
    print(f"Model loaded.", flush=True)

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = args.max_tasks or task_suite.n_tasks
    print(f"Task suite: {args.task_suite}, {num_tasks} tasks, {args.num_trials} trials each", flush=True)

    # Collect rollouts
    all_hidden_states = []
    all_actions = []
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
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": LIBERO_ENV_RESOLUTION, "camera_widths": LIBERO_ENV_RESOLUTION}
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)
        task_description = task.language

        for trial_idx in range(min(args.num_trials, len(initial_states))):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx])
            action_plan = collections.deque()

            episode_hidden_states = []
            episode_actions = []
            done = False
            t = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Build observation
                obs_dict, raw_state = build_observation_dict(obs, task_description)

                # Extract hidden state
                h = extract_features_from_dict(model, obs_dict)  # [1, 2048]
                episode_hidden_states.append(h[0])

                if not action_plan:
                    # Get new action chunk from model
                    observation = _model.Observation.from_dict(obs_dict)
                    rng, sample_rng = jax.random.split(rng)
                    action_chunk = sample_actions_jit(sample_rng, observation)
                    action_chunk = np.asarray(action_chunk[0])  # [action_horizon, action_dim]
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                episode_actions.append(action[:7].astype(np.float32))  # first 7 dims

                obs, reward, done, info = env.step(action[:7].tolist())
                if done:
                    total_successes += 1
                    break
                t += 1

            total_episodes += 1
            is_success = bool(done)

            # Store episode data
            ep_len = len(episode_hidden_states)
            for step_idx in range(ep_len):
                all_hidden_states.append(episode_hidden_states[step_idx])
                all_actions.append(episode_actions[step_idx] if step_idx < len(episode_actions) else np.zeros(7, dtype=np.float32))
                all_episode_ids.append(episode_counter)
                all_task_ids.append(task_id)
                all_timesteps.append(step_idx)
                all_is_success.append(is_success)

            episode_counter += 1

            if total_episodes % 5 == 0:
                print(f"  Episodes: {total_episodes}, Successes: {total_successes} "
                      f"({total_successes/total_episodes*100:.1f}%), "
                      f"Task {task_id+1}/{num_tasks}", flush=True)

        env.close()

    # Save
    print(f"\nTotal: {total_episodes} episodes, {total_successes} successes "
          f"({total_successes/total_episodes*100:.1f}%)", flush=True)
    print(f"Total hidden states: {len(all_hidden_states)}", flush=True)

    save_path = output_dir / f"rollouts_{args.task_suite}.npz"
    np.savez_compressed(
        save_path,
        hidden_states=np.stack(all_hidden_states),
        actions=np.stack(all_actions),
        episode_ids=np.array(all_episode_ids),
        task_ids=np.array(all_task_ids),
        timesteps=np.array(all_timesteps),
        is_success=np.array(all_is_success),
    )
    print(f"Saved to {save_path}", flush=True)


if __name__ == "__main__":
    main()
