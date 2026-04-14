"""Train pairwise Bradley-Terry value function on LIBERO-90 rollout data.

Downloads rollout data from HF, builds within-task success/failure pairs,
trains value function, evaluates ranking accuracy, and re-evaluates
world models with the real scorer.

Usage:
    PYTHONPATH=/workspace/openpi/src /workspace/openpi/.venv/bin/python -u \
        scripts/run_train_value_function.py \
        --output-dir data/contact_mpc/value_function
"""

import argparse
import pathlib
import pickle

import numpy as np
import torch

print("Loading modules...", flush=True)

from huggingface_hub import hf_hub_download
from openpi.contact_mpc.value_function.pairwise_dataset import (
    build_within_task_pairs,
    split_held_out_tasks,
)
from openpi.contact_mpc.value_function.train import train_value_function


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="arif101/libero90_vlm_features")
    parser.add_argument("--output-dir", default="data/contact_mpc/value_function")
    parser.add_argument("--n-pairs", type=int, default=50000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollout data
    print("Downloading rollout data from HF...", flush=True)
    rollout_path = hf_hub_download(args.hf_repo, "rollouts_libero_90.npz", repo_type="dataset")
    d = np.load(rollout_path)

    hidden_states = d["hidden_states"]
    is_success = d["is_success"]
    task_ids = d["task_ids"]
    episode_ids = d["episode_ids"]

    n_success = is_success.sum()
    n_failure = (~is_success).sum()
    n_tasks = len(np.unique(task_ids))
    print(f"Loaded rollouts: {len(hidden_states)} decision points, "
          f"{n_success} success, {n_failure} failure, {n_tasks} tasks", flush=True)

    # Build within-task pairs
    print("\n=== Building within-task pairs ===", flush=True)
    h_success, h_failure, pair_task_ids = build_within_task_pairs(
        hidden_states, is_success, task_ids, episode_ids,
        n_pairs=args.n_pairs,
    )

    # Split into train and held-out-tasks
    splits = split_held_out_tasks(h_success, h_failure, pair_task_ids)

    # Train value function
    print("\n=== Training Value Function ===", flush=True)
    model, metrics = train_value_function(
        splits["train_success"],
        splits["train_failure"],
        val_success=splits["test_success"],
        val_failure=splits["test_failure"],
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Report held-out-tasks accuracy (R5 residual risk)
    held_out_acc = metrics["val_ranking_accuracy"]
    print(f"\nHeld-out-tasks ranking accuracy: {held_out_acc:.3f}", flush=True)

    # Also evaluate on within-task held-out episodes (train/test split on episodes, not tasks)
    # This uses the training tasks but different pairs
    print("\n=== Evaluating on training-task pairs ===", flush=True)
    model.eval()
    with torch.no_grad():
        train_s = torch.tensor(splits["train_success"][:5000], dtype=torch.float32)
        train_f = torch.tensor(splits["train_failure"][:5000], dtype=torch.float32)
        train_acc = (model(train_s) > model(train_f)).float().mean().item()
    print(f"Training-task ranking accuracy: {train_acc:.3f}", flush=True)

    # R5 check: gap between train-task and held-out-task accuracy
    gap = train_acc - held_out_acc
    print(f"\nTrain-task vs held-out-task gap: {gap:.3f}", flush=True)
    if gap > 0.15:
        print("WARNING: >15 point gap — value function may be task-memorizing (R5 active)", flush=True)
    else:
        print("Gap is acceptable — value function generalizes across tasks", flush=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "value_function.pt")
    torch.save({"input_dim": hidden_states.shape[1], "hidden_dim": 256}, output_dir / "value_function_config.pt")
    np.savez(output_dir / "value_function_metrics.npz", **{k: np.array(v) for k, v in metrics.items()})
    print(f"\nSaved value function to {output_dir}", flush=True)

    # Re-evaluate world models with the real value function
    print("\n=== Re-evaluating world models with value function ===", flush=True)
    wm_dir = pathlib.Path("data/contact_mpc/world_model")
    if wm_dir.exists():
        from openpi.contact_mpc.world_model.architecture import LatentWorldModel
        from openpi.contact_mpc.world_model.train import build_dataset_for_horizon
        from openpi.contact_mpc.features.dataset import FeatureDataset

        features_path = hf_hub_download(args.hf_repo, "libero90_features_H10.npz", repo_type="dataset")
        features = FeatureDataset.load(features_path)

        wm_files = sorted(wm_dir.glob("world_model_H*.pt"))
        wm_files = [f for f in wm_files if "config" not in f.name and "metrics" not in f.name]

        for wm_path in wm_files:
            tag = wm_path.stem.replace("world_model_", "")
            H = int(tag.split("_")[0][1:])
            try:
                config = torch.load(str(wm_path).replace("world_model_", "world_model_config_"), weights_only=False)
                wm = LatentWorldModel(config)
                wm.load_state_dict(torch.load(str(wm_path), weights_only=True))
                wm.eval()

                # Get early/late pairs from demo features
                ep_ids = features.episode_ids
                ts = features.timesteps
                ep_lens = {}
                for i in range(len(ep_ids)):
                    ep_lens.setdefault(int(ep_ids[i]), 0)
                    ep_lens[int(ep_ids[i])] = max(ep_lens[int(ep_ids[i])], int(ts[i]) + 1)

                h, a, fh = build_dataset_for_horizon(features, H)
                early = [i for i in range(len(ts[:len(h)])) if ts[i] / ep_lens[int(ep_ids[i])] < 0.5]
                late = [i for i in range(len(ts[:len(h)])) if ts[i] / ep_lens[int(ep_ids[i])] > 0.75]

                rng = np.random.RandomState(42)
                n = min(1000, len(early), len(late))
                ei = rng.choice(early, n, replace=True)
                li = rng.choice(late, n, replace=True)

                with torch.no_grad():
                    pred_early = wm(torch.tensor(h[ei], dtype=torch.float32), torch.tensor(a[ei], dtype=torch.float32))
                    pred_late = wm(torch.tensor(h[li], dtype=torch.float32), torch.tensor(a[li], dtype=torch.float32))
                    score_early = model(pred_early)
                    score_late = model(pred_late)
                    acc = (score_late > score_early).float().mean().item()

                status = "PASS" if acc >= 0.70 else "FAIL"
                print(f"  {tag}: KS3={acc:.3f} {status}", flush=True)
            except Exception as e:
                print(f"  {tag}: error — {e}", flush=True)
    else:
        print("No world models found. Train them first with run_train_world_model.py", flush=True)

    # Also test: can the value function rank success vs failure on RAW rollout hidden states?
    print("\n=== Direct ranking on rollout hidden states ===", flush=True)
    success_idx = np.where(is_success)[0]
    failure_idx = np.where(~is_success)[0]
    rng = np.random.RandomState(42)
    n = min(1000, len(success_idx), len(failure_idx))
    si = rng.choice(success_idx, n, replace=True)
    fi = rng.choice(failure_idx, n, replace=True)

    model.eval()
    with torch.no_grad():
        s_scores = model(torch.tensor(hidden_states[si], dtype=torch.float32))
        f_scores = model(torch.tensor(hidden_states[fi], dtype=torch.float32))
        direct_acc = (s_scores > f_scores).float().mean().item()
    print(f"Direct ranking accuracy (success vs failure rollout states): {direct_acc:.3f}", flush=True)
    if direct_acc >= 0.70:
        print("Value function can rank raw hidden states — MPC viable even without world model", flush=True)
    else:
        print("Value function struggles on raw states — world model predictions may be needed", flush=True)


if __name__ == "__main__":
    main()
