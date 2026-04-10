"""
Test waypoint conditioning on LIBERO-90 tasks.

Runs Pi0.5 with ground-truth state waypoints on tasks it has successful demos for.
Compares: with waypoints vs without waypoints.

Usage (Terminal 2, LIBERO Python 3.8 env):
    source /workspace/libero_venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:/workspace/openpi/third_party/libero
    MUJOCO_GL=egl python /workspace/openpi/scripts/test_waypoint_client.py \
        --waypoint-file /workspace/openpi/waypoints/waypoints_state.pkl \
        --num-tasks 5 \
        --num-trials 3
"""

import argparse
import collections
import logging
import math
import pathlib
import pickle
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--waypoint-file", required=True)
    parser.add_argument("--num-tasks", type=int, default=5,
                        help="Number of tasks to test (picks tasks with successful demos)")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def get_current_state(obs):
    """Extract robot state from LIBERO observation."""
    return np.concatenate((
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    )).astype(np.float32)


def pad_state_to_action_dim(state, action_dim=32):
    """Pad state vector to action_dim (Pi0.5 expects action_dim-sized inputs)."""
    padded = np.zeros(action_dim, dtype=np.float32)
    padded[:len(state)] = state
    return padded


def run_episode(client, env, task_description, args, waypoints=None):
    """Run one episode, optionally with waypoint conditioning.

    Args:
        waypoints: list of [8] state vectors (keyframe targets). If None, run without conditioning.

    Returns:
        success: bool
        steps_taken: int
    """
    action_plan = collections.deque()
    t = 0

    # Waypoint tracking
    current_waypoint_idx = 0
    waypoint_threshold = 0.15  # L2 distance to consider waypoint reached

    while t < args.max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            state = np.concatenate((
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))

            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }

                # Add target state if waypoints provided
                if waypoints is not None and current_waypoint_idx < len(waypoints):
                    target = waypoints[current_waypoint_idx]
                    element["target_state"] = pad_state_to_action_dim(target)

                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[:args.replan_steps])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())

            # Check if current waypoint is reached
            if waypoints is not None and current_waypoint_idx < len(waypoints):
                current_state = get_current_state(obs)
                target = waypoints[current_waypoint_idx]
                dist = np.linalg.norm(current_state[:3] - target[:3])  # xyz distance
                if dist < waypoint_threshold:
                    logging.info(f"    Waypoint {current_waypoint_idx} reached (dist={dist:.4f})")
                    current_waypoint_idx += 1

            if done:
                return True, t
            t += 1

        except Exception as e:
            logging.error(f"Error: {e}")
            break

    return False, t


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load waypoint data
    logging.info(f"Loading waypoints from {args.waypoint_file}")
    with open(args.waypoint_file, "rb") as f:
        all_waypoints = pickle.load(f)

    # Find unique tasks with waypoint data
    task_map = {}
    for wp in all_waypoints:
        desc = wp["task_description"]
        if desc not in task_map:
            task_map[desc] = wp
    logging.info(f"Found waypoints for {len(task_map)} unique tasks")

    # Connect to server
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_90"]()
    num_tasks = task_suite.n_tasks

    # Find matching tasks
    tasks_to_test = []
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        desc = task.language
        if desc in task_map:
            tasks_to_test.append((task_id, task, desc, task_map[desc]))
        if len(tasks_to_test) >= args.num_tasks:
            break

    logging.info(f"Testing {len(tasks_to_test)} tasks with waypoint data")

    # Run evaluation
    results_with = {"successes": 0, "total": 0}
    results_without = {"successes": 0, "total": 0}

    for task_id, task, desc, wp_data in tasks_to_test:
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        waypoints = wp_data["keyframe_states"]  # [N_kf, 8]

        logging.info(f"\nTask: {desc}")
        logging.info(f"  Waypoints: {len(waypoints)} keyframes")

        for trial in range(args.num_trials):
            # Run WITHOUT waypoints (baseline)
            env.reset()
            obs = env.set_init_state(initial_states[trial % len(initial_states)])
            success_without, steps_without = run_episode(
                client, env, task_description, args, waypoints=None
            )
            results_without["total"] += 1
            if success_without:
                results_without["successes"] += 1

            # Run WITH waypoints
            env.reset()
            obs = env.set_init_state(initial_states[trial % len(initial_states)])
            success_with, steps_with = run_episode(
                client, env, task_description, args, waypoints=waypoints[1:]  # skip first (start state)
            )
            results_with["total"] += 1
            if success_with:
                results_with["successes"] += 1

            logging.info(
                f"  Trial {trial+1}: without={success_without} ({steps_without} steps), "
                f"with={success_with} ({steps_with} steps)"
            )

        env.close()

    # Summary
    rate_without = results_without["successes"] / max(results_without["total"], 1)
    rate_with = results_with["successes"] / max(results_with["total"], 1)

    logging.info("\n" + "=" * 60)
    logging.info("WAYPOINT CONDITIONING TEST RESULTS")
    logging.info("=" * 60)
    logging.info(f"Tasks tested: {len(tasks_to_test)}")
    logging.info(f"Trials per task: {args.num_trials}")
    logging.info(f"Without waypoints: {results_without['successes']}/{results_without['total']} = {rate_without:.1%}")
    logging.info(f"With waypoints:    {results_with['successes']}/{results_with['total']} = {rate_with:.1%}")
    logging.info(f"Difference:        {rate_with - rate_without:+.1%}")

    if rate_with > rate_without:
        logging.info("\nWaypoint conditioning HELPS. Proceed to building the prediction head.")
    elif rate_with == rate_without:
        logging.info("\nNo difference. The target_state_proj layer is untrained (random weights).")
        logging.info("This is expected — the model doesn't know how to use the waypoint token yet.")
        logging.info("Need to train the target_state_proj on successful demo segments.")
    else:
        logging.info("\nWaypoint conditioning HURTS. The random waypoint token is adding noise.")
        logging.info("This is expected with untrained projection. Need training before conclusions.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
