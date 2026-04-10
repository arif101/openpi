"""
Extract STATE-BASED keyframe waypoints from successful LIBERO-90 rollouts.

Uses robot proprioceptive state [7] (xyz + rotation + gripper) as waypoints
instead of SigLIP image embeddings. No model loading needed — just reads
the state vectors from the rollout data.

Usage:
    cd /openpi
    uv run python scripts/extract_waypoints_state.py \
        --hf-dataset arif101/libero90_successes \
        --output-file /openpi/waypoints/waypoints_state.pkl
"""

import argparse
import json
import logging
import math
import os
import pathlib
import pickle
import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-file", type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--vel-threshold", type=float, default=-0.1)
    parser.add_argument("--min-keyframe-gap", type=int, default=2)
    return parser.parse_args()


def load_from_lerobot(repo_id):
    """Load rollouts from HuggingFace LeRobot dataset."""
    logging.info(f"Loading LeRobot dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    logging.info(f"Dataset has {len(dataset)} frames, {dataset.num_episodes} episodes")

    rollouts = []
    for ep_idx in range(dataset.num_episodes):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()

        trajectory = []
        task_description = None

        for frame_idx in range(ep_start, ep_end):
            frame = dataset[frame_idx]

            state = frame["state"]
            if hasattr(state, "numpy"):
                state = state.numpy()

            actions = frame["actions"]
            if hasattr(actions, "numpy"):
                actions = actions.numpy()
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]

            if task_description is None:
                task_description = frame.get("task", "unknown task")
                if hasattr(task_description, "item"):
                    task_description = task_description.item()

            trajectory.append({
                "state": state.astype(np.float32),
                "actions": actions.astype(np.float32),
                "prompt": str(task_description),
            })

        rollouts.append({
            "task_id": ep_idx,
            "task_description": str(task_description),
            "trajectory": trajectory,
            "num_steps": len(trajectory),
        })

        if (ep_idx + 1) % 50 == 0:
            logging.info(f"  Loaded {ep_idx + 1}/{dataset.num_episodes} episodes")

    logging.info(f"Loaded {len(rollouts)} rollouts")
    return rollouts


def load_rollouts(args):
    if args.rollout_file:
        logging.info(f"Loading from {args.rollout_file}")
        with open(args.rollout_file, "rb") as f:
            return pickle.load(f)
    elif args.hf_dataset:
        return load_from_lerobot(args.hf_dataset)
    else:
        raise ValueError("Must provide --rollout-file or --hf-dataset")


def extract_keyframe_indices(trajectory, vel_threshold=-0.1, min_gap=2):
    """Find keyframe indices based on gripper and velocity changes."""
    keyframes = [0]

    for i in range(1, len(trajectory)):
        if i - keyframes[-1] < min_gap:
            continue

        curr_actions = trajectory[i]["actions"]
        prev_actions = trajectory[i - 1]["actions"]

        curr_gripper = curr_actions[0, -1] if curr_actions.ndim == 2 else curr_actions[-1]
        prev_gripper = prev_actions[0, -1] if prev_actions.ndim == 2 else prev_actions[-1]
        gripper_changed = (curr_gripper > 0) != (prev_gripper > 0)

        curr_vel = curr_actions[0, :6] if curr_actions.ndim == 2 else curr_actions[:6]
        prev_vel = prev_actions[0, :6] if prev_actions.ndim == 2 else prev_actions[:6]

        curr_norm = np.linalg.norm(curr_vel)
        prev_norm = np.linalg.norm(prev_vel)
        if curr_norm > 1e-6 and prev_norm > 1e-6:
            dot = np.dot(curr_vel / curr_norm, prev_vel / prev_norm)
            direction_changed = dot < vel_threshold
        else:
            direction_changed = False

        if gripper_changed or direction_changed:
            keyframes.append(i)

    if keyframes[-1] != len(trajectory) - 1:
        keyframes.append(len(trajectory) - 1)

    return keyframes


def main():
    args = parse_args()

    output_path = pathlib.Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total, used, free = shutil.disk_usage(output_path.parent)
    free_gb = free / (1024 ** 3)
    logging.info(f"Disk: {free_gb:.1f}GB free")

    rollouts = load_rollouts(args)

    logging.info("Extracting keyframes and state waypoints...")
    all_waypoint_data = []
    total_keyframes = 0
    total_segments = 0
    keyframe_counts = []

    for i, rollout in enumerate(rollouts):
        traj = rollout["trajectory"]
        kf_indices = extract_keyframe_indices(
            traj, vel_threshold=args.vel_threshold, min_gap=args.min_keyframe_gap,
        )
        keyframe_counts.append(len(kf_indices))
        total_keyframes += len(kf_indices)

        # State waypoints at each keyframe
        kf_states = np.stack([traj[idx]["state"] for idx in kf_indices])  # [N_kf, 7]

        # Action segments between keyframes
        segments = []
        for j in range(len(kf_indices) - 1):
            start_idx = kf_indices[j]
            end_idx = kf_indices[j + 1]
            segment_actions = []
            segment_states = []
            for step_idx in range(start_idx, end_idx):
                segment_actions.append(traj[step_idx]["actions"])
                segment_states.append(traj[step_idx]["state"])
            segments.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_state": kf_states[j],       # [7]
                "end_state": kf_states[j + 1],      # [7]
                "actions": segment_actions,
                "states": segment_states,
                "num_steps": end_idx - start_idx,
            })
            total_segments += 1

        all_waypoint_data.append({
            "rollout_idx": i,
            "task_id": rollout["task_id"],
            "task_description": rollout["task_description"],
            "prompt": traj[0]["prompt"],
            "keyframe_indices": kf_indices,
            "keyframe_states": kf_states,           # [N_kf, 7]
            "num_keyframes": len(kf_indices),
            "trajectory_length": len(traj),
            "segments": segments,
        })

        if (i + 1) % 50 == 0:
            logging.info(f"  Processed {i + 1}/{len(rollouts)} rollouts")

    # Save
    logging.info(f"Saving to {args.output_file}")
    with open(args.output_file, "wb") as f:
        pickle.dump(all_waypoint_data, f)

    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)

    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("STATE WAYPOINT EXTRACTION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Rollouts: {len(rollouts)}")
    logging.info(f"Total keyframes: {total_keyframes}")
    logging.info(f"Avg keyframes/rollout: {np.mean(keyframe_counts):.1f} (min={np.min(keyframe_counts)}, max={np.max(keyframe_counts)})")
    logging.info(f"Total segments: {total_segments}")
    logging.info(f"Waypoint dim: 7 (xyz:3 + rotation:3 + gripper:1)")
    logging.info(f"File size: {file_size_mb:.1f} MB")

    # Sanity check: state distance stats
    logging.info("\nState waypoint sanity check:")

    # Distance between adjacent keyframes (same rollout)
    adjacent_dists = []
    for wp in all_waypoint_data[:50]:
        states = wp["keyframe_states"]
        for j in range(len(states) - 1):
            dist = np.linalg.norm(states[j] - states[j + 1])
            adjacent_dists.append(dist)

    # Distance between random pairs
    all_states = np.concatenate([wp["keyframe_states"] for wp in all_waypoint_data])
    random_dists = []
    for _ in range(200):
        a, b = np.random.choice(len(all_states), 2, replace=False)
        dist = np.linalg.norm(all_states[a] - all_states[b])
        random_dists.append(dist)

    logging.info(f"  Adjacent keyframe distance: {np.mean(adjacent_dists):.4f} ± {np.std(adjacent_dists):.4f}")
    logging.info(f"  Random pair distance:       {np.mean(random_dists):.4f} ± {np.std(random_dists):.4f}")
    logging.info(f"  Ratio:                      {np.mean(random_dists)/np.mean(adjacent_dists):.2f}x "
                 f"(should be >1 — random pairs are farther apart)")

    # Per-dimension stats
    logging.info("\nPer-dimension range across all keyframe states:")
    dim_names = ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"]
    for d in range(min(7, all_states.shape[1])):
        logging.info(f"  {dim_names[d]:8s}: min={all_states[:, d].min():.4f}, "
                     f"max={all_states[:, d].max():.4f}, "
                     f"range={all_states[:, d].max() - all_states[:, d].min():.4f}")

    # Gripper state distribution at keyframes
    gripper_dim = min(6, all_states.shape[1] - 1)
    gripper_vals = all_states[:, gripper_dim]
    n_open = np.sum(gripper_vals > 0)
    n_closed = np.sum(gripper_vals <= 0)
    logging.info(f"\nGripper at keyframes: {n_open} open, {n_closed} closed "
                 f"({n_open/len(gripper_vals)*100:.0f}% / {n_closed/len(gripper_vals)*100:.0f}%)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
