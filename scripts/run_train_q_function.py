"""Train the action-conditional Q(h, a) value function.

Replaces the legacy ``run_train_value_function.py`` (which trained V(h)) for
the LIBERO-90 robustness work. Three improvements over V(h):

1. ``Q(h, a)`` differentiates candidates by the action chunk that led to the
   leaf — V(h) couldn't, which is part of why MCTS leaves with similar
   post-action latents collapsed to ties.
2. Multi-suite training: load rollout npz files from multiple LIBERO suites
   and concatenate. Reduces the 19-task memorization that broke V(h) on
   LIBERO-90 perturbed.
3. Feature-space perturbation augmentation: add Gaussian noise scaled to
   per-dim std during training, simulating the eval-time object-position
   jitter. ``--perturb-noise-std-fraction`` knob.

Reads ``rollouts_<suite>.npz`` files (produced by ``run_collect_rollouts.py``),
which have keys: hidden_states, action_chunks, episode_ids, task_ids,
timesteps, is_success. Both successes AND failures are needed; demo-only
FeatureDataset npz files are not enough because they have 0 failure pairs.

Usage:
    PYTHONPATH=/workspace/openpi/src /workspace/openpi/.venv/bin/python -u \\
        scripts/run_train_q_function.py \\
        --rollouts-files \\
            arif101/libero90_vlm_features:rollouts_libero_90.npz \\
        --output-dir data/contact_mpc/q_function

Add more suites by passing additional ``hf_repo:filename`` entries.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import torch

print("Loading modules...", flush=True)

from huggingface_hub import hf_hub_download

from openpi.contact_mpc.value_function.pairwise_dataset import (
    build_within_task_pairs_qha,
    split_held_out_tasks_qha,
)
from openpi.contact_mpc.value_function.train import train_q_function

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class RolloutBundle:
    """Minimal rollout npz schema: hidden_states + action_chunks + labels."""
    hidden_states: np.ndarray
    action_chunks: np.ndarray
    episode_ids: np.ndarray
    task_ids: np.ndarray
    is_success: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rollouts-files",
        nargs="+",
        default=["arif101/libero90_vlm_features:rollouts_libero_90.npz"],
        help="One or more 'hf_repo:filename' entries pointing to rollout .npz "
             "files (produced by run_collect_rollouts.py).",
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


def _load_one(spec: str) -> RolloutBundle:
    """Load a rollout npz from a 'hf_repo:filename' spec."""
    if ":" not in spec:
        raise ValueError(f"--rollouts-files entry must be 'hf_repo:filename', got: {spec}")
    repo, filename = spec.split(":", 1)
    print(f"Downloading {filename} from {repo}...", flush=True)
    path = hf_hub_download(repo, filename, repo_type="dataset")
    d = np.load(path)
    return RolloutBundle(
        hidden_states=d["hidden_states"],
        action_chunks=d["action_chunks"],
        episode_ids=d["episode_ids"],
        task_ids=d["task_ids"],
        is_success=d["is_success"],
    )


def _concat(bundles: list[RolloutBundle]) -> RolloutBundle:
    """Concatenate rollout bundles, offsetting episode/task ids for uniqueness."""
    if len(bundles) == 1:
        return bundles[0]

    h_list, a_list = [], []
    ep_list, tid_list, success_list = [], [], []
    ep_offset, tid_offset = 0, 0
    for b in bundles:
        h_list.append(b.hidden_states)
        a_list.append(b.action_chunks)
        ep_list.append(b.episode_ids + ep_offset)
        tid_list.append(b.task_ids + tid_offset)
        success_list.append(b.is_success)
        ep_offset = int(ep_list[-1].max()) + 1
        tid_offset = int(tid_list[-1].max()) + 1

    return RolloutBundle(
        hidden_states=np.concatenate(h_list),
        action_chunks=np.concatenate(a_list),
        episode_ids=np.concatenate(ep_list),
        task_ids=np.concatenate(tid_list),
        is_success=np.concatenate(success_list),
    )


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate all rollout bundles
    print(f"\n=== Loading {len(args.rollouts_files)} rollout files ===", flush=True)
    bundles = [_load_one(s) for s in args.rollouts_files]
    for spec, b in zip(args.rollouts_files, bundles):
        n_succ = int(b.is_success.sum())
        n_fail = int((~b.is_success).sum())
        n_tasks = len(np.unique(b.task_ids))
        print(
            f"  {spec}: {len(b.hidden_states)} records, "
            f"{n_succ} success / {n_fail} failure, {n_tasks} tasks",
            flush=True,
        )

    bundle = _concat(bundles)
    print(
        f"\nCombined: {len(bundle.hidden_states)} records, "
        f"{int(bundle.is_success.sum())} success / "
        f"{int((~bundle.is_success).sum())} failure, "
        f"{len(np.unique(bundle.task_ids))} tasks",
        flush=True,
    )

    # Build (h, a, success/failure) pairs within each task
    print("\n=== Building within-task Q(h, a) pairs ===", flush=True)
    pairs = build_within_task_pairs_qha(
        hidden_states=bundle.hidden_states,
        action_chunks=bundle.action_chunks,
        is_success=bundle.is_success,
        task_ids=bundle.task_ids,
        episode_ids=bundle.episode_ids,
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
        "hidden_state_dim": bundle.hidden_states.shape[1],
        "action_chunk_horizon": bundle.action_chunks.shape[1],
        "action_dim": bundle.action_chunks.shape[2],
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
