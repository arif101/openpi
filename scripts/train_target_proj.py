"""
Train the target_state_proj layer so Pi0.5's action expert learns to use
the waypoint conditioning token.

Method:
  For each segment (start_state → end_state) in the waypoint data:
    1. Load the observation at the start of the segment
    2. Set target_state = end_state (the next waypoint)
    3. Run compute_loss with target_state conditioning
    4. Backprop through target_state_proj only (everything else frozen)

This teaches the projection layer to encode target states in a way the
action expert can use to generate trajectories toward the target.

Usage:
    cd /workspace/openpi
    uv run python scripts/train_target_proj.py \
        --waypoint-file /workspace/openpi/waypoints/waypoints_state.pkl \
        --hf-dataset arif101/libero90_successes \
        --num-steps 1000 \
        --lr 1e-3 \
        --save-dir /workspace/target_proj_checkpoint
"""

import argparse
import logging
import os
import pathlib
import pickle
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.shared import download as _download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoint-file", required=True)
    parser.add_argument("--hf-dataset", default="arif101/libero90_successes")
    parser.add_argument("--checkpoint-dir", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", default="/workspace/target_proj_checkpoint")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def load_hf_images(hf_dataset_id, episode_idx, frame_indices):
    """Load specific frames from the HuggingFace dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(hf_dataset_id)

    ep_start = dataset.episode_data_index["from"][episode_idx].item()
    frames = []
    for fi in frame_indices:
        frame = dataset[ep_start + fi]
        img = frame["image"]
        if hasattr(img, "numpy"):
            img = img.numpy()
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        wrist = frame.get("wrist_image", img)
        if hasattr(wrist, "numpy"):
            wrist = wrist.numpy()
        if wrist.dtype == np.float32 or wrist.dtype == np.float64:
            wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
        if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
            wrist = np.transpose(wrist, (1, 2, 0))

        state = frame["state"]
        if hasattr(state, "numpy"):
            state = state.numpy()

        actions = frame["actions"]
        if hasattr(actions, "numpy"):
            actions = actions.numpy()

        frames.append({
            "image": img,
            "wrist_image": wrist,
            "state": state.astype(np.float32),
            "actions": actions.astype(np.float32),
        })
    return frames


def build_training_examples(waypoint_data, hf_dataset_id):
    """Build training examples from waypoint segments.

    Each example: (observation_at_segment_start, actions_in_segment, target_end_state)
    """
    logging.info("Building training examples from waypoint segments...")
    logging.info(f"Loading images from HuggingFace: {hf_dataset_id}")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(hf_dataset_id)

    examples = []
    for wp_idx, wp in enumerate(waypoint_data):
        ep_idx = wp["rollout_idx"]
        ep_start = dataset.episode_data_index["from"][ep_idx].item()

        for seg in wp["segments"]:
            start_idx = seg["start_idx"]

            # Load observation at segment start
            frame = dataset[ep_start + start_idx]

            img = frame["image"]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))

            wrist = frame.get("wrist_image", img)
            if hasattr(wrist, "numpy"):
                wrist = wrist.numpy()
            if wrist.dtype == np.float32 or wrist.dtype == np.float64:
                wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
            if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
                wrist = np.transpose(wrist, (1, 2, 0))

            state = frame["state"]
            if hasattr(state, "numpy"):
                state = state.numpy()

            actions = frame["actions"]
            if hasattr(actions, "numpy"):
                actions = actions.numpy()

            # Target state = end of this segment (next waypoint)
            target_state = seg["end_state"]

            examples.append({
                "image": img.astype(np.float32) / 127.5 - 1.0,  # normalize to [-1, 1]
                "wrist_image": wrist.astype(np.float32) / 127.5 - 1.0,
                "state": state.astype(np.float32),
                "actions": actions.astype(np.float32),
                "target_state": target_state.astype(np.float32),
                "prompt": wp["prompt"],
            })

        if (wp_idx + 1) % 50 == 0:
            logging.info(f"  Processed {wp_idx + 1}/{len(waypoint_data)} rollouts")

    logging.info(f"Built {len(examples)} training examples from {len(waypoint_data)} rollouts")
    return examples


def build_training_examples_fast(waypoint_data, dataset):
    """Build training examples from pre-loaded LeRobot dataset."""
    logging.info("Building training examples...")

    examples = []
    for wp_idx, wp in enumerate(waypoint_data):
        ep_idx = wp["rollout_idx"]
        ep_start = dataset.episode_data_index["from"][ep_idx].item()

        for seg in wp["segments"]:
            start_idx = seg["start_idx"]
            frame = dataset[ep_start + start_idx]

            img = frame["image"]
            if hasattr(img, "numpy"):
                img = img.numpy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))

            wrist = frame.get("wrist_image", img)
            if hasattr(wrist, "numpy"):
                wrist = wrist.numpy()
            if wrist.dtype == np.float32 or wrist.dtype == np.float64:
                wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
            if wrist.ndim == 3 and wrist.shape[0] in (1, 3):
                wrist = np.transpose(wrist, (1, 2, 0))

            state = frame["state"]
            if hasattr(state, "numpy"):
                state = state.numpy()

            actions = frame["actions"]
            if hasattr(actions, "numpy"):
                actions = actions.numpy()

            examples.append({
                "image": img.astype(np.float32) / 127.5 - 1.0,
                "wrist_image": wrist.astype(np.float32) / 127.5 - 1.0,
                "state": state.astype(np.float32),
                "actions": actions.astype(np.float32),
                "target_state": seg["end_state"].astype(np.float32),
                "prompt": wp["prompt"],
            })

        if (wp_idx + 1) % 50 == 0:
            logging.info(f"  Processed {wp_idx + 1}/{len(waypoint_data)} rollouts ({len(examples)} examples)")

    logging.info(f"Built {len(examples)} training examples")
    return examples


def pad_to_action_dim(arr, action_dim=32):
    """Pad array to action_dim."""
    if len(arr) >= action_dim:
        return arr[:action_dim]
    padded = np.zeros(action_dim, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


def main():
    args = parse_args()

    # Load waypoint data
    logging.info(f"Loading waypoints from {args.waypoint_file}")
    with open(args.waypoint_file, "rb") as f:
        waypoint_data = pickle.load(f)
    logging.info(f"Loaded {len(waypoint_data)} rollouts with waypoints")

    # Pre-load the HuggingFace dataset once (slow, but only once)
    logging.info(f"Loading HuggingFace dataset: {args.hf_dataset}")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    hf_dataset = LeRobotDataset(args.hf_dataset)
    logging.info(f"Dataset loaded: {len(hf_dataset)} frames, {hf_dataset.num_episodes} episodes")

    # Build training examples using pre-loaded dataset
    examples = build_training_examples_fast(waypoint_data, hf_dataset)
    if not examples:
        logging.error("No training examples! Exiting.")
        return

    # Load model
    logging.info("Loading Pi0.5 model...")
    train_config = _config.get_config("pi05_libero")
    checkpoint_dir = _download.maybe_download(str(args.checkpoint_dir))
    params = _model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)

    # Initialize target_state_proj with random weights (not in checkpoint)
    logging.info("Initializing target_state_proj with random weights...")
    rng = jax.random.key(42)
    action_dim = train_config.model.action_dim  # 32
    expert_width = 1024  # gemma_300m action expert width
    k1, k2 = jax.random.split(rng)
    # Kaiming uniform init for kernel, zeros for bias
    kernel_init = jax.random.normal(k1, (action_dim, expert_width)) * (2.0 / action_dim) ** 0.5
    bias_init = jnp.zeros(expert_width)
    model.target_state_proj.kernel.value = kernel_init.astype(jnp.bfloat16)
    model.target_state_proj.bias.value = bias_init.astype(jnp.bfloat16)
    logging.info(f"  kernel: {model.target_state_proj.kernel.value.shape}")
    logging.info(f"  bias: {model.target_state_proj.bias.value.shape}")
    logging.info("Model loaded")

    # Set up optimizer for target_state_proj ONLY
    # Find the target_state_proj params
    target_proj_filter = nnx_utils.PathRegex(".*target_state_proj.*")
    target_proj_params = nnx.state(model, target_proj_filter)
    n_params = sum(np.prod(p.value.shape) for p in jax.tree.leaves(target_proj_params)
                   if hasattr(p, 'value') and hasattr(p.value, 'shape'))
    logging.info(f"Training target_state_proj: {n_params} parameters")

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(target_proj_params)

    # Training loop
    logging.info(f"Starting training for {args.num_steps} steps...")
    losses = []
    start_time = time.time()

    for step in range(args.num_steps):
        # Sample random example
        ex = examples[np.random.randint(len(examples))]

        # Build observation (single example, add batch dim)
        # Pad state and actions to action_dim=32
        state_padded = pad_to_action_dim(ex["state"])
        target_padded = pad_to_action_dim(ex["target_state"])
        actions_padded = np.zeros((train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32)
        if ex["actions"].ndim == 1:
            actions_padded[0, :len(ex["actions"])] = ex["actions"]
        else:
            ah = min(ex["actions"].shape[0], train_config.model.action_horizon)
            ad = min(ex["actions"].shape[1], train_config.model.action_dim)
            actions_padded[:ah, :ad] = ex["actions"][:ah, :ad]

        # Create observation dict matching model's expected format
        observation = _model.Observation(
            images={
                "base_0_rgb": jnp.array(ex["image"])[np.newaxis],
                "left_wrist_0_rgb": jnp.array(ex["wrist_image"])[np.newaxis],
                "right_wrist_0_rgb": jnp.zeros_like(jnp.array(ex["image"]))[np.newaxis],
            },
            image_masks={
                "base_0_rgb": jnp.array([True]),
                "left_wrist_0_rgb": jnp.array([True]),
                "right_wrist_0_rgb": jnp.array([False]),
            },
            state=jnp.array(state_padded)[np.newaxis],
            tokenized_prompt=None,
            tokenized_prompt_mask=None,
        )
        actions = jnp.array(actions_padded)[np.newaxis]
        target_state = jnp.array(target_padded)[np.newaxis]

        # Compute loss and gradient for target_state_proj only
        rng = jax.random.key(step)

        diff_state = nnx.DiffState(0, target_proj_filter)

        def loss_fn(model, rng, obs, acts, ts):
            chunked_loss = model.compute_loss(rng, obs, acts, train=True, target_state=ts)
            return jnp.mean(chunked_loss)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model, rng, observation, actions, target_state
        )

        # Update target_state_proj params
        target_proj_params = nnx.state(model, target_proj_filter)
        updates, opt_state = optimizer.update(grads, opt_state, target_proj_params)
        new_params = optax.apply_updates(target_proj_params, updates)
        nnx.update(model, new_params)

        losses.append(float(loss))

        if (step + 1) % args.log_interval == 0:
            avg_loss = np.mean(losses[-args.log_interval:])
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            logging.info(
                f"Step {step + 1}/{args.num_steps}: loss={avg_loss:.4f}, "
                f"rate={rate:.1f} steps/s, elapsed={elapsed:.0f}s"
            )

    # Save the trained target_state_proj weights
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_proj_params = nnx.state(model, target_proj_filter)
    params_dict = {k: np.array(v.value) for k, v in
                   jax.tree_util.tree_leaves_with_path(target_proj_params)}

    save_path = save_dir / "target_state_proj.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(params_dict, f)

    logging.info(f"\nTraining complete!")
    logging.info(f"Final loss: {np.mean(losses[-50:]):.4f}")
    logging.info(f"Initial loss: {np.mean(losses[:50]):.4f}")
    logging.info(f"Saved target_state_proj weights to {save_path}")
    logging.info(f"Total time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
