"""Train the action-conditional Q(h, a) value function.

Replaces the legacy ``run_train_value_function.py`` (which trained V(h)) for
the LIBERO-90 robustness work. Three improvements over V(h):

1. ``Q(h, a)`` differentiates candidates by the action chunk that led to the
   leaf — V(h) couldn't, which is part of why MCTS leaves with similar
   post-action latents collapsed to ties.
2. Multi-suite training: load FeatureDataset npz files from multiple LIBERO
   suites and concatenate. Reduces the 19-task memorization that broke
   V(h) on LIBERO-90 perturbed.
3. Feature-space perturbation augmentation: add Gaussian noise scaled to
   per-dim std during training, simulating the eval-time object-position
   jitter. ``--perturb-noise-std-fraction`` knob.

Usage:
    PYTHONPATH=/workspace/openpi/src /workspace/openpi/.venv/bin/python -u \\
        scripts/run_train_q_function.py \\
        --features-files \\
            arif101/libero90_vlm_features:libero90_features_H10.npz \\
        --output-dir data/contact_mpc/q_function

Add more suites by passing additional ``hf_repo:filename`` entries.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
import torch

print("Loading modules...", flush=True)

from huggingface_hub import hf_hub_download

from openpi.contact_mpc.features.dataset import FeatureDataset
from openpi.contact_mpc.value_function.pairwise_dataset import (
    build_within_task_pairs_qha,
    split_held_out_tasks_qha,
)
from openpi.contact_mpc.value_function.train import train_q_function

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features-files",
        nargs="+",
        default=["arif101/libero90_vlm_features:libero90_features_H10.npz"],
        help="One or more 'hf_repo:filename' entries pointing to FeatureDataset .npz files.",
    )
    p.add_argument("--output-dir", default="data/contact_mpc/q_function")
    p.add_argument("--n-pairs", type=int, default=50000)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--action-emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--perturb-noise-std-fraction",
        type=float,
        default=0.1,
        help="Fraction of per-dim std added as Gaussian noise to hidden states each batch. "
             "0 disables. ~0.1 simulates a moderate (~5cm) perturbation in feature space.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_one(spec: str) -> FeatureDataset:
    """Load a FeatureDataset from a 'hf_repo:filename' spec."""
    if ":" not in spec:
        raise ValueError(f"--features-files entry must be 'hf_repo:filename', got: {spec}")
    repo, filename = spec.split(":", 1)
    print(f"Downloading {filename} from {repo}...", flush=True)
    path = hf_hub_download(repo, filename, repo_type="dataset")
    return FeatureDataset.load(path)


def _concat(datasets: list[FeatureDataset]) -> FeatureDataset:
    """Concatenate FeatureDatasets, offsetting episode/task ids to preserve uniqueness."""
    if len(datasets) == 1:
        return datasets[0]

    h_list, a_list, fh_list = [], [], []
    ep_list, tid_list, ts_list = [], [], []
    contact_list, success_list = [], []

    ep_offset, tid_offset = 0, 0
    horizon = datasets[0].horizon
    for ds in datasets:
        if ds.horizon != horizon:
            raise ValueError(
                f"Cannot concat FeatureDatasets with different horizons: "
                f"{horizon} vs {ds.horizon}"
            )
        h_list.append(ds.hidden_states)
        a_list.append(ds.action_chunks)
        fh_list.append(ds.future_hidden_states)
        ep_list.append(ds.episode_ids + ep_offset)
        tid_list.append(ds.task_ids + tid_offset)
        ts_list.append(ds.timesteps)
        contact_list.append(ds.is_contact)
        success_list.append(ds.is_success)
        ep_offset = int(ep_list[-1].max()) + 1
        tid_offset = int(tid_list[-1].max()) + 1

    return FeatureDataset(
        hidden_states=np.concatenate(h_list),
        action_chunks=np.concatenate(a_list),
        future_hidden_states=np.concatenate(fh_list),
        episode_ids=np.concatenate(ep_list),
        task_ids=np.concatenate(tid_list),
        timesteps=np.concatenate(ts_list),
        is_contact=np.concatenate(contact_list),
        is_success=np.concatenate(success_list),
        horizon=horizon,
    )


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate all FeatureDataset files
    print(f"\n=== Loading {len(args.features_files)} feature files ===", flush=True)
    datasets = [_load_one(s) for s in args.features_files]
    for spec, ds in zip(args.features_files, datasets):
        n_succ = int(ds.is_success.sum())
        n_fail = int((~ds.is_success).sum())
        n_tasks = len(np.unique(ds.task_ids))
        print(
            f"  {spec}: {len(ds.hidden_states)} records, "
            f"{n_succ} success / {n_fail} failure, {n_tasks} tasks",
            flush=True,
        )

    features = _concat(datasets)
    print(
        f"\nCombined: {len(features.hidden_states)} records, "
        f"{int(features.is_success.sum())} success / "
        f"{int((~features.is_success).sum())} failure, "
        f"{len(np.unique(features.task_ids))} tasks",
        flush=True,
    )

    # Build (h, a, success/failure) pairs within each task
    print("\n=== Building within-task Q(h, a) pairs ===", flush=True)
    pairs = build_within_task_pairs_qha(
        hidden_states=features.hidden_states,
        action_chunks=features.action_chunks,
        is_success=features.is_success,
        task_ids=features.task_ids,
        episode_ids=features.episode_ids,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )
    splits = split_held_out_tasks_qha(pairs)

    # Train Q(h, a)
    print("\n=== Training Q(h, a) ===", flush=True)
    model, metrics = train_q_function(
        h_success=splits["train_h_success"],
        a_success=splits["train_a_success"],
        h_failure=splits["train_h_failure"],
        a_failure=splits["train_a_failure"],
        val_h_success=splits["test_h_success"],
        val_a_success=splits["test_a_success"],
        val_h_failure=splits["test_h_failure"],
        val_a_failure=splits["test_a_failure"],
        action_emb_dim=args.action_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        perturb_noise_std_fraction=args.perturb_noise_std_fraction,
        seed=args.seed,
    )

    # Hold-out-tasks accuracy is exactly the val_ranking_accuracy from train_q_function
    held_out_acc = metrics["val_ranking_accuracy"]
    print(f"\nHeld-out-tasks ranking accuracy: {held_out_acc:.3f}", flush=True)

    # Within-task accuracy on training pairs (for memorization gap diagnostic)
    print("\n=== Evaluating on training-task pairs ===", flush=True)
    model.eval()
    with torch.no_grad():
        n_eval = min(5000, len(splits["train_h_success"]))
        train_hs = torch.tensor(splits["train_h_success"][:n_eval], dtype=torch.float32)
        train_as = torch.tensor(splits["train_a_success"][:n_eval], dtype=torch.float32)
        train_hf = torch.tensor(splits["train_h_failure"][:n_eval], dtype=torch.float32)
        train_af = torch.tensor(splits["train_a_failure"][:n_eval], dtype=torch.float32)
        train_acc = (model(train_hs, train_as) > model(train_hf, train_af)).float().mean().item()
    print(f"Training-task ranking accuracy: {train_acc:.3f}", flush=True)

    gap = train_acc - held_out_acc
    print(f"\nTrain-task vs held-out-task gap: {gap:.3f}", flush=True)
    if gap > 0.15:
        print(
            "WARNING: >15 point gap — Q-function may still be task-memorizing. "
            "Try larger --perturb-noise-std-fraction, more --features-files, "
            "or higher --weight-decay.",
            flush=True,
        )
    else:
        print("Gap is acceptable — Q-function generalizes across tasks", flush=True)

    # Save model + config + metrics
    cfg = {
        "type": "ActionConditionalValueFunction",
        "hidden_state_dim": features.hidden_states.shape[1],
        "action_chunk_horizon": features.action_chunks.shape[1],
        "action_dim": features.action_chunks.shape[2],
        "action_emb_dim": args.action_emb_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    torch.save(model.state_dict(), output_dir / "q_function.pt")
    torch.save(cfg, output_dir / "q_function_config.pt")
    np.savez(
        output_dir / "q_function_metrics.npz",
        **{
            k: np.array(v) for k, v in {
                **metrics,
                "train_task_accuracy": train_acc,
                "memorization_gap": gap,
            }.items()
        },
    )
    print(f"\nSaved Q(h, a) to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
