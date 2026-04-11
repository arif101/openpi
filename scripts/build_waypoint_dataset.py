"""
Build a new LeRobot dataset with target_state column from the existing
libero90_successes dataset.

For each frame at index t, target_state[t] = state at the NEXT keyframe
(after t). For frames after the last keyframe, target_state = state at
the last keyframe (the goal).

Keyframes are detected by gripper state changes and velocity direction
reversals — same logic as scripts/extract_waypoints_state.py.

Usage:
    cd /workspace/openpi
    uv run python scripts/build_waypoint_dataset.py \
        --source-dataset arif101/libero90_successes \
        --output-repo-id arif101/libero90_waypoints \
        --push-to-hub
"""

import argparse
import logging
import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", default="arif101/libero90_successes")
    parser.add_argument("--output-repo-id", default="arif101/libero90_waypoints")
    parser.add_argument("--vel-threshold", type=float, default=-0.1)
    parser.add_argument("--min-keyframe-gap", type=int, default=2)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def extract_keyframe_indices(actions_seq, vel_threshold=-0.1, min_gap=2):
    """Find keyframe indices from a sequence of action vectors.

    Detects:
    - gripper state changes (action dim -1 sign flip)
    - velocity direction reversals (action dims 0:6 dot product < threshold)

    Args:
        actions_seq: list/array of [action_dim] vectors, one per frame
    Returns:
        sorted list of keyframe indices (always includes 0 and last index)
    """
    keyframes = [0]
    for i in range(1, len(actions_seq)):
        if i - keyframes[-1] < min_gap:
            continue

        curr = actions_seq[i]
        prev = actions_seq[i - 1]

        # Gripper change: sign flip on last dim
        gripper_changed = (curr[-1] > 0) != (prev[-1] > 0)

        # Velocity direction reversal: normalized dot product on first 6 dims
        curr_vel = curr[:6]
        prev_vel = prev[:6]
        curr_norm = np.linalg.norm(curr_vel)
        prev_norm = np.linalg.norm(prev_vel)
        if curr_norm > 1e-6 and prev_norm > 1e-6:
            dot = np.dot(curr_vel / curr_norm, prev_vel / prev_norm)
            direction_changed = dot < vel_threshold
        else:
            direction_changed = False

        if gripper_changed or direction_changed:
            keyframes.append(i)

    if keyframes[-1] != len(actions_seq) - 1:
        keyframes.append(len(actions_seq) - 1)

    return keyframes


def compute_target_states(states_seq, keyframe_indices):
    """For each frame, return the state at the NEXT keyframe (after this frame).

    For frames at or beyond the last keyframe, target = state at last keyframe.
    """
    n = len(states_seq)
    target_states = np.zeros_like(states_seq)
    for t in range(n):
        # Find the smallest keyframe index strictly greater than t
        next_kf = None
        for kf in keyframe_indices:
            if kf > t:
                next_kf = kf
                break
        if next_kf is None:
            # No more keyframes — use last keyframe (the goal state)
            next_kf = keyframe_indices[-1]
        target_states[t] = states_seq[next_kf]
    return target_states


def to_numpy(x):
    """Convert tensor or array to numpy."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def main():
    args = parse_args()

    logging.info(f"Loading source dataset: {args.source_dataset}")
    source = LeRobotDataset(args.source_dataset)
    logging.info(f"  {len(source)} frames, {source.num_episodes} episodes")

    # Probe shapes from first frame
    sample_frame = source[0]
    sample_state = to_numpy(sample_frame["state"])
    sample_actions = to_numpy(sample_frame["actions"])
    sample_image = to_numpy(sample_frame["image"])
    sample_wrist = to_numpy(sample_frame.get("wrist_image", sample_frame["image"]))

    # Normalize image to HWC uint8
    if sample_image.ndim == 3 and sample_image.shape[0] in (1, 3):
        sample_image = np.transpose(sample_image, (1, 2, 0))
    if sample_image.dtype != np.uint8:
        sample_image = (sample_image * 255).clip(0, 255).astype(np.uint8)

    state_dim = sample_state.shape[-1]
    action_dim = sample_actions.shape[-1] if sample_actions.ndim == 1 else sample_actions.shape[-1]
    img_h, img_w = sample_image.shape[:2]
    logging.info(f"  state_dim={state_dim}, action_dim={action_dim}, image={img_h}x{img_w}")

    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / args.output_repo_id
    if output_path.exists():
        logging.info(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create output dataset
    logging.info(f"Creating output dataset: {args.output_repo_id}")
    output = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (img_h, img_w, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "target_state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["target_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    total_keyframes = 0
    for ep_idx in range(source.num_episodes):
        ep_start = source.episode_data_index["from"][ep_idx].item()
        ep_end = source.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start

        # Load episode states and actions
        states_list = []
        actions_list = []
        task_desc = None
        for fi in range(ep_start, ep_end):
            frame = source[fi]
            states_list.append(to_numpy(frame["state"]).astype(np.float32))
            a = to_numpy(frame["actions"]).astype(np.float32)
            if a.ndim > 1:
                a = a[0]  # collapse if stored as [horizon, action_dim]
            actions_list.append(a)
            if task_desc is None:
                td = frame.get("task", "unknown")
                task_desc = td.item() if hasattr(td, "item") else str(td)
        states_arr = np.stack(states_list)
        actions_arr = np.stack(actions_list)

        # Compute keyframes from action sequence
        keyframes = extract_keyframe_indices(
            actions_arr, vel_threshold=args.vel_threshold, min_gap=args.min_keyframe_gap,
        )
        total_keyframes += len(keyframes)

        # Compute target_state for every frame
        targets = compute_target_states(states_arr, keyframes)

        # Add frames to output dataset
        for fi_local in range(ep_len):
            fi = ep_start + fi_local
            frame = source[fi]

            img = to_numpy(frame["image"])
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)

            wrist = to_numpy(frame.get("wrist_image", frame["image"]))
            if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
                wrist = np.transpose(wrist, (1, 2, 0))
            if wrist.dtype != np.uint8:
                wrist = (wrist * 255).clip(0, 255).astype(np.uint8)

            output.add_frame({
                "image": img,
                "wrist_image": wrist,
                "state": states_arr[fi_local],
                "target_state": targets[fi_local].astype(np.float32),
                "actions": actions_arr[fi_local].astype(np.float32),
                "task": task_desc,
            })

        output.save_episode()

        if (ep_idx + 1) % 25 == 0:
            logging.info(f"  Converted {ep_idx + 1}/{source.num_episodes} episodes")

    logging.info(f"\nDataset built: {len(output)} frames, {output.num_episodes} episodes")
    logging.info(f"Total keyframes: {total_keyframes} (avg {total_keyframes/source.num_episodes:.1f}/episode)")
    logging.info(f"Saved to: {output_path}")

    if args.push_to_hub:
        logging.info(f"Pushing to HuggingFace Hub: {args.output_repo_id}")
        output.push_to_hub(
            tags=["libero", "libero_90", "panda", "waypoints"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
