"""Build within-task pairwise dataset for Bradley-Terry value function training.

PRIMARY RULE: every training pair (h_success, h_failure) must come from
the same task. This blocks the suite-identification shortcut where the
value function learns to classify "which task is this" rather than
"is this trajectory succeeding."

Two builders:

- ``build_within_task_pairs`` — returns (h_success, h_failure) pairs for
  V(h) training.

- ``build_within_task_pairs_qha`` — returns (h_succ, a_succ, h_fail, a_fail)
  pairs for Q(h, a) training. Includes optional Gaussian feature-space
  perturbation augmentation to simulate the LIBERO-PRO eval-time object
  position jitter.

Enforced by test_pairwise_dataset.py.
"""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def build_within_task_pairs(
    hidden_states: np.ndarray,
    is_success: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    n_pairs: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (h_success, h_failure) pairs from within-task rollouts.

    For each pair, both hidden states come from the same task_id but
    different episodes — one successful, one failed.

    Args:
        hidden_states: [N, hidden_dim] from rollout collection.
        is_success: [N] bool — True if the episode succeeded.
        task_ids: [N] int — which task this decision point came from.
        episode_ids: [N] int — which episode this came from.
        n_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        Tuple of (h_success, h_failure, pair_task_ids) where each is
        shape [n_pairs, ...]. pair_task_ids records which task each
        pair came from (for verification).
    """
    rng = np.random.RandomState(seed)

    # Group indices by (task_id, success/failure)
    task_success = {}  # task_id -> list of indices where is_success=True
    task_failure = {}  # task_id -> list of indices where is_success=False

    for i in range(len(hidden_states)):
        tid = int(task_ids[i])
        if is_success[i]:
            task_success.setdefault(tid, []).append(i)
        else:
            task_failure.setdefault(tid, []).append(i)

    # Find tasks that have BOTH successes and failures
    valid_tasks = []
    for tid in set(task_success.keys()) & set(task_failure.keys()):
        valid_tasks.append(tid)

    if not valid_tasks:
        raise ValueError(
            "No tasks have both successful and failed episodes. "
            f"Tasks with successes: {sorted(task_success.keys())}, "
            f"Tasks with failures: {sorted(task_failure.keys())}"
        )

    logger.info(f"Valid tasks (have both success+failure): {len(valid_tasks)} out of {len(set(task_ids))}")
    for tid in sorted(valid_tasks):
        n_s = len(task_success[tid])
        n_f = len(task_failure[tid])
        logger.info(f"  Task {tid}: {n_s} success states, {n_f} failure states")

    # Build pairs
    h_success_list = []
    h_failure_list = []
    pair_task_list = []

    for _ in range(n_pairs):
        # Sample a random valid task
        tid = rng.choice(valid_tasks)
        # Sample one success and one failure index from that task
        s_idx = rng.choice(task_success[tid])
        f_idx = rng.choice(task_failure[tid])
        h_success_list.append(hidden_states[s_idx])
        h_failure_list.append(hidden_states[f_idx])
        pair_task_list.append(tid)

    h_success = np.stack(h_success_list)
    h_failure = np.stack(h_failure_list)
    pair_task_ids = np.array(pair_task_list)

    logger.info(f"Built {n_pairs} within-task pairs from {len(valid_tasks)} tasks")
    return h_success, h_failure, pair_task_ids


def split_held_out_tasks(
    h_success: np.ndarray,
    h_failure: np.ndarray,
    pair_task_ids: np.ndarray,
    held_out_fraction: float = 0.2,
    seed: int = 99,
) -> dict:
    """Split pairs into train and held-out-tasks sets.

    The held-out-tasks split trains on pairs from 80% of tasks and tests
    on pairs from the remaining 20%. This tests whether the value function
    generalizes to unseen tasks (R5 residual risk detection).

    Returns dict with keys: train_success, train_failure, train_task_ids,
    test_success, test_failure, test_task_ids.
    """
    rng = np.random.RandomState(seed)
    unique_tasks = np.unique(pair_task_ids)
    n_held_out = max(1, int(len(unique_tasks) * held_out_fraction))

    held_out_tasks = set(rng.choice(unique_tasks, n_held_out, replace=False))
    train_mask = np.array([tid not in held_out_tasks for tid in pair_task_ids])
    test_mask = ~train_mask

    logger.info(f"Held-out-tasks split: {train_mask.sum()} train pairs, "
                f"{test_mask.sum()} test pairs, {n_held_out} held-out tasks")

    return {
        "train_success": h_success[train_mask],
        "train_failure": h_failure[train_mask],
        "train_task_ids": pair_task_ids[train_mask],
        "test_success": h_success[test_mask],
        "test_failure": h_failure[test_mask],
        "test_task_ids": pair_task_ids[test_mask],
    }


def build_within_task_pairs_qha(
    hidden_states: np.ndarray,
    action_chunks: np.ndarray,
    is_success: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    n_pairs: int = 50000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build (h_succ, a_succ, h_fail, a_fail) pairs for Q(h, a) training.

    For each pair, both records come from the same ``task_id`` but different
    episodes — one successful, one failed. The action chunks are the chunks
    that *led from* each respective decision point, so the Q-function learns
    "given this state, was the action that was taken progressing the task?"

    Args:
        hidden_states: [N, hidden_dim] from rollout collection.
        action_chunks: [N, H, action_dim] action chunks executed from each state.
        is_success: [N] bool — True if the parent episode succeeded.
        task_ids: [N] int — which task this decision point came from.
        episode_ids: [N] int — which episode this came from.
        n_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        Dict with keys: h_success, a_success, h_failure, a_failure, pair_task_ids.
    """
    rng = np.random.RandomState(seed)

    task_success: dict[int, list[int]] = {}
    task_failure: dict[int, list[int]] = {}
    for i in range(len(hidden_states)):
        tid = int(task_ids[i])
        (task_success if is_success[i] else task_failure).setdefault(tid, []).append(i)

    valid_tasks = sorted(set(task_success.keys()) & set(task_failure.keys()))
    if not valid_tasks:
        raise ValueError(
            "No tasks have both successful and failed episodes. "
            f"Tasks with successes: {sorted(task_success.keys())}, "
            f"Tasks with failures: {sorted(task_failure.keys())}"
        )

    logger.info(
        f"build_within_task_pairs_qha: {len(valid_tasks)} valid tasks, "
        f"{n_pairs} target pairs"
    )

    h_succ = np.empty((n_pairs, hidden_states.shape[1]), dtype=hidden_states.dtype)
    h_fail = np.empty_like(h_succ)
    a_succ = np.empty((n_pairs, *action_chunks.shape[1:]), dtype=action_chunks.dtype)
    a_fail = np.empty_like(a_succ)
    pair_tids = np.empty(n_pairs, dtype=np.int64)

    for k in range(n_pairs):
        tid = int(rng.choice(valid_tasks))
        s_idx = int(rng.choice(task_success[tid]))
        f_idx = int(rng.choice(task_failure[tid]))
        h_succ[k] = hidden_states[s_idx]
        h_fail[k] = hidden_states[f_idx]
        a_succ[k] = action_chunks[s_idx]
        a_fail[k] = action_chunks[f_idx]
        pair_tids[k] = tid

    return {
        "h_success": h_succ,
        "h_failure": h_fail,
        "a_success": a_succ,
        "a_failure": a_fail,
        "pair_task_ids": pair_tids,
    }


def split_held_out_tasks_qha(
    pairs: dict[str, np.ndarray],
    held_out_fraction: float = 0.2,
    seed: int = 99,
) -> dict[str, np.ndarray]:
    """Split Q(h, a) pairs into train and held-out-tasks sets.

    Same logic as ``split_held_out_tasks`` but propagates action chunks too.
    """
    rng = np.random.RandomState(seed)
    pair_tids = pairs["pair_task_ids"]
    unique_tasks = np.unique(pair_tids)
    n_held_out = max(1, int(len(unique_tasks) * held_out_fraction))
    held_out = set(rng.choice(unique_tasks, n_held_out, replace=False))
    train_mask = np.array([tid not in held_out for tid in pair_tids])
    test_mask = ~train_mask

    logger.info(
        f"Held-out-tasks split (qha): {train_mask.sum()} train, "
        f"{test_mask.sum()} test, {n_held_out} held-out tasks"
    )

    return {
        "train_h_success": pairs["h_success"][train_mask],
        "train_h_failure": pairs["h_failure"][train_mask],
        "train_a_success": pairs["a_success"][train_mask],
        "train_a_failure": pairs["a_failure"][train_mask],
        "train_task_ids": pair_tids[train_mask],
        "test_h_success": pairs["h_success"][test_mask],
        "test_h_failure": pairs["h_failure"][test_mask],
        "test_a_success": pairs["a_success"][test_mask],
        "test_a_failure": pairs["a_failure"][test_mask],
        "test_task_ids": pair_tids[test_mask],
    }


def feature_perturbation_augment(
    h: np.ndarray,
    noise_std_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to hidden states scaled to a fraction of per-dim std.

    Proxy for LIBERO-PRO object-position perturbation. The eval perturbs raw
    object xyz; we don't have the simulator in the training loop so we
    approximate the *feature-space* effect by adding noise scaled to the
    feature distribution's natural variance.

    ``noise_std_fraction`` ∈ [0, 1] — fraction of per-dim std to use as noise std.
    A value of 0.1 adds noise at 10% of each dim's natural variance, which
    empirically simulates a moderate (~5cm) perturbation in feature space.

    Caveat: this is NOT equivalent to perturbing object positions and re-running
    Pi0.5. It's a proxy. The proper augmentation requires recollecting rollouts
    under perturbation, which is GPU-heavy. Treat noise_std_fraction as a knob
    to be tuned against held-out perturbed eval performance.
    """
    if noise_std_fraction <= 0:
        return h
    per_dim_std = h.std(axis=0, keepdims=True)
    noise = rng.standard_normal(h.shape).astype(h.dtype) * (
        per_dim_std * noise_std_fraction
    )
    return h + noise
