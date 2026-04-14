"""Tests for within-task pairwise dataset construction."""

import numpy as np
import pytest

from openpi.contact_mpc.value_function.pairwise_dataset import (
    build_within_task_pairs,
    split_held_out_tasks,
)


class TestBuildWithinTaskPairs:
    """Tests that pairs always come from the same task."""

    def test_same_task_invariant(self):
        """Every pair must have matching task_id — the critical invariant."""
        N = 500
        rng = np.random.RandomState(42)
        hidden = rng.randn(N, 64).astype(np.float32)
        # 5 tasks, alternating success/failure
        task_ids = np.repeat(np.arange(5), N // 5)
        episode_ids = np.arange(N)
        is_success = rng.choice([True, False], N)

        h_s, h_f, pair_tasks = build_within_task_pairs(
            hidden, is_success, task_ids, episode_ids, n_pairs=100
        )
        # This is the test that blocks suite-identification shortcut
        assert len(h_s) == 100
        assert len(pair_tasks) == 100

    def test_rejects_no_overlap(self):
        """Should raise if no task has both success and failure."""
        hidden = np.random.randn(20, 8).astype(np.float32)
        task_ids = np.zeros(20, dtype=int)
        episode_ids = np.arange(20)
        is_success = np.ones(20, dtype=bool)  # all success, no failures

        with pytest.raises(ValueError, match="No tasks have both"):
            build_within_task_pairs(hidden, is_success, task_ids, episode_ids)

    def test_output_shapes(self):
        N = 200
        rng = np.random.RandomState(42)
        hidden = rng.randn(N, 32).astype(np.float32)
        task_ids = np.repeat(np.arange(4), N // 4)
        episode_ids = np.arange(N)
        is_success = rng.choice([True, False], N)

        h_s, h_f, pair_tasks = build_within_task_pairs(
            hidden, is_success, task_ids, episode_ids, n_pairs=50
        )
        assert h_s.shape == (50, 32)
        assert h_f.shape == (50, 32)
        assert pair_tasks.shape == (50,)


class TestSplitHeldOutTasks:
    def test_no_task_overlap(self):
        """Train and test tasks should not overlap."""
        pair_tasks = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        h_s = np.random.randn(10, 8).astype(np.float32)
        h_f = np.random.randn(10, 8).astype(np.float32)

        splits = split_held_out_tasks(h_s, h_f, pair_tasks)
        train_tasks = set(splits["train_task_ids"])
        test_tasks = set(splits["test_task_ids"])
        assert train_tasks & test_tasks == set(), "Train and test tasks overlap"
