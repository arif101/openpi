"""Day 2: Extract VLM hidden states from libero90_successes and run linear probe.

Usage (on GPU machine):
    cd /workspace/openpi
    uv run python scripts/run_extract_features.py \
        --dataset arif101/libero90_successes \
        --output-dir data/contact_mpc/features \
        --horizon 10 \
        --run-probe
"""

import argparse
import logging
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.contact_mpc.features.dataset import FeatureDataset, detect_contact_timesteps
from openpi.contact_mpc.features.extractor import extract_features_from_dict, get_hidden_dim
from openpi.contact_mpc.features.probe import evaluate_ranking_accuracy, train_linear_probe, train_mlp_probe
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
from openpi.training import config as train_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="arif101/libero90_successes")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    parser.add_argument("--output-dir", default="data/contact_mpc/features")
    parser.add_argument("--horizon", type=int, default=10, help="Action chunk horizon H")
    parser.add_argument("--run-probe", action="store_true", help="Run linear probe after extraction")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes (for debugging)")
    return parser.parse_args()


def load_pi05_model(checkpoint_path: str):
    """Load frozen Pi0.5 model from checkpoint."""
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
    logger.info(f"Loaded Pi0.5 model, hidden_dim={get_hidden_dim(model)}")
    return model


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_pi05_model(args.checkpoint)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    ds = LeRobotDataset(args.dataset)
    logger.info(f"  {len(ds)} frames, {ds.num_episodes} episodes")

    num_episodes = args.max_episodes or ds.num_episodes
    H = args.horizon

    # Extract features episode by episode
    all_hidden_states = []
    all_action_chunks = []
    all_future_hidden = []
    all_episode_ids = []
    all_task_ids = []
    all_timesteps = []
    all_is_contact = []
    all_is_success = []

    for ep_idx in range(num_episodes):
        ep_start = ds.episode_data_index["from"][ep_idx].item()
        ep_end = ds.episode_data_index["to"][ep_idx].item()
        ep_len = ep_end - ep_start

        if ep_len <= H:
            continue

        # Collect episode actions for contact detection
        actions_list = []
        task_id = 0  # Default; update if task_index available
        for fi in range(ep_start, ep_end):
            frame = ds[fi]
            a = np.asarray(frame["actions"]).astype(np.float32)
            if a.ndim > 1:
                a = a[0]
            actions_list.append(a)
            if "task_index" in frame:
                task_id = int(frame["task_index"])
        actions_arr = np.stack(actions_list)

        # Detect contact timesteps
        contact_mask = detect_contact_timesteps(actions_arr)

        # Extract hidden states for each timestep in this episode
        # Process in batches to avoid OOM
        ep_hidden_states = []
        batch_size = 8
        for batch_start in range(0, ep_len, batch_size):
            batch_end = min(batch_start + batch_size, ep_len)
            batch_frames = [ds[ep_start + t] for t in range(batch_start, batch_end)]

            # Build batched observation dict
            # This follows the pattern from libero_policy.py
            images = np.stack([np.asarray(f["image"]).astype(np.uint8) for f in batch_frames])
            wrist_images = np.stack([
                np.asarray(f.get("wrist_image", f["image"])).astype(np.uint8)
                for f in batch_frames
            ])
            states = np.stack([np.asarray(f["state"]).astype(np.float32) for f in batch_frames])

            # Pad state to action_dim (32) as the model expects
            if states.shape[-1] < 32:
                states = np.pad(states, [(0, 0), (0, 32 - states.shape[-1])])

            # Ensure images are HWC uint8
            if images.ndim == 4 and images.shape[1] in (1, 3):
                images = np.transpose(images, (0, 2, 3, 1))
            if wrist_images.ndim == 4 and wrist_images.shape[1] in (1, 3):
                wrist_images = np.transpose(wrist_images, (0, 2, 3, 1))

            data = {
                "image": {
                    "base_0_rgb": images,
                    "left_wrist_0_rgb": wrist_images,
                    "right_wrist_0_rgb": np.zeros_like(images),
                },
                "image_mask": {
                    "base_0_rgb": np.ones(len(batch_frames), dtype=bool),
                    "left_wrist_0_rgb": np.ones(len(batch_frames), dtype=bool),
                    "right_wrist_0_rgb": np.ones(len(batch_frames), dtype=bool),
                },
                "state": states,
            }

            h = extract_features_from_dict(model, data)
            ep_hidden_states.append(h)

        ep_hidden = np.concatenate(ep_hidden_states, axis=0)  # [ep_len, hidden_dim]

        # Build (h_t, action_chunk, h_{t+H}) triples
        for t in range(ep_len - H):
            all_hidden_states.append(ep_hidden[t])
            all_action_chunks.append(actions_arr[t : t + H])
            all_future_hidden.append(ep_hidden[t + H])
            all_episode_ids.append(ep_idx)
            all_task_ids.append(task_id)
            all_timesteps.append(t)
            all_is_contact.append(contact_mask[t])
            all_is_success.append(True)  # demo dataset is all successes

        if (ep_idx + 1) % 25 == 0:
            logger.info(f"  Processed {ep_idx + 1}/{num_episodes} episodes")

    # Build dataset
    dataset = FeatureDataset(
        hidden_states=np.stack(all_hidden_states),
        action_chunks=np.stack(all_action_chunks),
        future_hidden_states=np.stack(all_future_hidden),
        episode_ids=np.array(all_episode_ids),
        task_ids=np.array(all_task_ids),
        timesteps=np.array(all_timesteps),
        is_contact=np.array(all_is_contact),
        is_success=np.array(all_is_success),
        horizon=H,
    )

    save_path = output_dir / f"libero90_features_H{H}.npz"
    dataset.save(str(save_path))
    logger.info(f"Dataset: {len(dataset.hidden_states)} triples, "
                f"hidden_dim={dataset.hidden_states.shape[1]}, "
                f"contact_fraction={dataset.is_contact.mean():.3f}")

    # Run linear probe if requested
    if args.run_probe:
        logger.info("\n=== Linear Probe (Representation Check) ===")
        # For the demo dataset, all labels are success=True.
        # We use temporal position as a proxy: hidden states near the end of
        # successful episodes should look different from early states.
        # A more meaningful probe requires failure data (Week 2).
        # For now, report the probe on temporal progress (early vs late in episode).
        ep_lengths = {}
        for i, (ep_id, t) in enumerate(zip(dataset.episode_ids, dataset.timesteps)):
            ep_lengths.setdefault(int(ep_id), 0)
            ep_lengths[int(ep_id)] = max(ep_lengths[int(ep_id)], int(t) + 1)

        # Label: 1 if in the last 25% of the episode (near goal), 0 otherwise
        progress_labels = np.array([
            1 if dataset.timesteps[i] > 0.75 * ep_lengths[int(dataset.episode_ids[i])] else 0
            for i in range(len(dataset.timesteps))
        ])

        if len(np.unique(progress_labels)) >= 2:
            # 1. Linear probe (most conservative test)
            logger.info("\n--- Linear Probe ---")
            linear_model, linear_acc = train_linear_probe(
                dataset.hidden_states, progress_labels
            )
            logger.info(f"Linear probe accuracy: {linear_acc:.3f}")

            # 2. MLP probe (fallback if linear fails)
            logger.info("\n--- MLP Probe (2-layer, 256 hidden) ---")
            mlp_model, mlp_acc = train_mlp_probe(
                dataset.hidden_states, progress_labels
            )
            logger.info(f"MLP probe accuracy: {mlp_acc:.3f}")

            # Interpretation
            logger.info("\n=== Interpretation ===")
            if linear_acc > 0.65:
                logger.info(f"LINEAR PROBE PASSES ({linear_acc:.3f} > 0.65). "
                            "Signal is linearly separable — strong result.")
                best_model = linear_model
            elif mlp_acc > 0.65:
                logger.info(f"Linear probe fails ({linear_acc:.3f}) but MLP PASSES ({mlp_acc:.3f} > 0.65). "
                            "Signal exists but needs nonlinear decoding — still viable for our 2-layer value function.")
                best_model = mlp_model
            else:
                logger.warning(f"BOTH PROBES FAIL (linear={linear_acc:.3f}, mlp={mlp_acc:.3f}). "
                               "Hidden states may not encode task progress. "
                               "Consider: different pooling strategy, per-token features, or deeper probe. "
                               "The search hypothesis is in trouble.")
                best_model = mlp_model  # save anyway for inspection

            logger.info("(Real success/failure probe needs Week 2 rollout data with actual failures)")

            # Save the best probe for use as KS3 proxy scorer
            import pickle
            probe_path = output_dir / "best_probe.pkl"
            with open(probe_path, "wb") as f:
                pickle.dump(best_model, f)
            logger.info(f"Saved best probe to {probe_path}")
        else:
            logger.warning("Not enough label diversity for probe — skipping")


if __name__ == "__main__":
    main()
