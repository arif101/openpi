"""Tests for DPO pair construction."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from openpi.contact_mpc.dpo.pair_builder import (
    DPOPair,
    build_pairs_for_cluster,
    load_pairs,
    save_pairs,
)


def _toy_rollouts(hidden_dim: int = 8, action_dim: int = 7, chunk_H: int = 5):
    """Synthesize a small rollout set with known structure:

    Task 0: episode 0 (success), episode 1 (failure), episode 2 (failure)
    Task 1: episode 3 (success), episode 4 (failure)
    Task 2: episode 5 (success only — no pairable failures)

    Each episode has 4 decision points at timesteps 10, 15, 20, 25.
    """
    rng = np.random.default_rng(0)
    decisions_per_episode = 4
    episode_structure = [
        (0, 0, True),   # task_id, episode_id, is_success
        (0, 1, False),
        (0, 2, False),
        (1, 3, True),
        (1, 4, False),
        (2, 5, True),
    ]

    hidden_states, action_chunks = [], []
    episode_ids, task_ids, timesteps, is_success = [], [], [], []
    for task_id, ep_id, succ in episode_structure:
        for d in range(decisions_per_episode):
            hidden_states.append(rng.standard_normal(hidden_dim).astype(np.float32))
            action_chunks.append(rng.standard_normal((chunk_H, action_dim)).astype(np.float32))
            episode_ids.append(ep_id)
            task_ids.append(task_id)
            timesteps.append(10 + d * 5)
            is_success.append(succ)

    return {
        "hidden_states": np.stack(hidden_states),
        "action_chunks": np.stack(action_chunks),
        "episode_ids": np.array(episode_ids),
        "task_ids": np.array(task_ids),
        "timesteps": np.array(timesteps),
        "is_success": np.array(is_success, dtype=bool),
    }


class TestBuildPairsForCluster:
    def test_yields_within_task_pairs_only(self):
        d = _toy_rollouts()
        pairs = build_pairs_for_cluster(
            cluster_id="c0",
            cluster_task_ids=[0],
            cluster_failure_episode_ids=[1, 2],
            **d,
            n_pairs_per_task=100,
            timestep_window=100,  # very permissive
        )
        assert all(p.task_id == 0 for p in pairs)
        # All preferred eps are successes on task 0 -> episode 0
        assert all(p.preferred_episode_id == 0 for p in pairs)
        # All rejected eps are failures on task 0 -> episodes 1 or 2
        assert all(p.rejected_episode_id in {1, 2} for p in pairs)

    def test_respects_cluster_failure_membership(self):
        d = _toy_rollouts()
        # Only episode 1 is in this cluster; episode 2 should not appear
        pairs = build_pairs_for_cluster(
            cluster_id="c0",
            cluster_task_ids=[0],
            cluster_failure_episode_ids=[1],
            **d,
            n_pairs_per_task=100,
            timestep_window=100,
        )
        assert pairs, "Expected some pairs"
        assert all(p.rejected_episode_id == 1 for p in pairs)

    def test_skips_tasks_with_no_success_or_no_failure(self):
        d = _toy_rollouts()
        # Task 2 has successes but no failures
        pairs = build_pairs_for_cluster(
            cluster_id="c_any",
            cluster_task_ids=[2],
            cluster_failure_episode_ids=[99],  # doesn't exist
            **d,
            n_pairs_per_task=10,
        )
        assert pairs == []

    def test_timestep_window_restricts_pairs(self):
        d = _toy_rollouts()
        # With window=0, only exact-timestep matches. Each decision has exactly
        # one counterpart at the same timestep (10,15,20,25) across the 1 success
        # and 2 failures: |S|=4, |F|=8, so 4 * 2 = 8 max pairs.
        pairs = build_pairs_for_cluster(
            cluster_id="c0",
            cluster_task_ids=[0],
            cluster_failure_episode_ids=[1, 2],
            **d,
            n_pairs_per_task=100,
            timestep_window=0,
        )
        for p in pairs:
            assert p.preferred_timestep == p.rejected_timestep

    def test_subsamples_to_n_pairs_per_task(self):
        d = _toy_rollouts()
        pairs = build_pairs_for_cluster(
            cluster_id="c0",
            cluster_task_ids=[0],
            cluster_failure_episode_ids=[1, 2],
            **d,
            n_pairs_per_task=3,
            timestep_window=100,
        )
        assert len(pairs) <= 3

    def test_multi_task_cluster(self):
        d = _toy_rollouts()
        pairs = build_pairs_for_cluster(
            cluster_id="c_multi",
            cluster_task_ids=[0, 1],
            cluster_failure_episode_ids=[1, 2, 4],
            **d,
            n_pairs_per_task=5,
            timestep_window=100,
        )
        tids = {p.task_id for p in pairs}
        assert tids == {0, 1}


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        d = _toy_rollouts()
        pairs = build_pairs_for_cluster(
            cluster_id="c0",
            cluster_task_ids=[0],
            cluster_failure_episode_ids=[1, 2],
            **d,
            n_pairs_per_task=5,
            timestep_window=100,
        )
        assert pairs, "Setup failure"
        p_out = tmp_path / "pairs.npz"
        save_pairs(pairs, str(p_out))
        loaded = load_pairs(str(p_out))
        assert len(loaded) == len(pairs)
        # Per-pair equivalence on hashable fields
        for a, b in zip(pairs, loaded):
            assert a.task_id == b.task_id
            assert a.cluster_id == b.cluster_id
            assert a.preferred_episode_id == b.preferred_episode_id
            np.testing.assert_array_equal(a.preferred_hidden_state, b.preferred_hidden_state)
            np.testing.assert_array_equal(a.rejected_action_chunk, b.rejected_action_chunk)

    def test_save_raises_on_empty(self, tmp_path):
        with pytest.raises(ValueError, match="No pairs"):
            save_pairs([], str(tmp_path / "x.npz"))
