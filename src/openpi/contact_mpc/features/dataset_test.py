"""Tests for contact detection and FeatureDataset."""

import tempfile

import numpy as np
import pytest

from openpi.contact_mpc.features.dataset import (
    FeatureDataset,
    detect_contact_timesteps,
)


class TestDetectContactTimesteps:
    """Tests for the gripper-based contact detection heuristic."""

    def test_no_contact_on_constant_actions(self):
        """Constant actions should produce no contact events."""
        actions = np.tile(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (20, 1))
        contacts = detect_contact_timesteps(actions)
        assert not contacts.any()

    def test_detects_gripper_sign_flip(self):
        """A gripper sign flip (open -> close) should be detected."""
        actions = np.zeros((10, 7))
        actions[:, -1] = 1.0  # gripper open
        actions[5:, -1] = -1.0  # gripper close at t=5
        contacts = detect_contact_timesteps(actions, min_gap=1)
        assert contacts[5], "Should detect contact at gripper sign flip"
        assert contacts.sum() == 1, "Should detect exactly one contact"

    def test_detects_velocity_reversal(self):
        """A velocity direction reversal should be detected."""
        actions = np.zeros((10, 7))
        # Moving forward then reversing
        actions[:5, 0] = 1.0  # forward
        actions[5:, 0] = -1.0  # backward (dot product = -1, well below threshold)
        actions[:, -1] = 1.0  # gripper stays open (no gripper change)
        contacts = detect_contact_timesteps(actions, min_gap=1)
        assert contacts[5], "Should detect velocity reversal at t=5"

    def test_min_gap_enforced(self):
        """Contacts closer than min_gap should be suppressed."""
        actions = np.zeros((10, 7))
        actions[:, -1] = 1.0
        # Two gripper flips, 2 steps apart
        actions[3, -1] = -1.0
        actions[4, -1] = 1.0  # flip back — only 1 step after previous
        actions[5, -1] = -1.0  # flip again — 2 steps after first
        contacts = detect_contact_timesteps(actions, min_gap=2)
        assert contacts[3], "First flip should be detected"
        assert not contacts[4], "Second flip should be suppressed by min_gap"

    def test_returns_correct_shape(self):
        actions = np.random.randn(50, 7)
        contacts = detect_contact_timesteps(actions)
        assert contacts.shape == (50,)
        assert contacts.dtype == bool

    def test_first_frame_never_contact(self):
        """Frame 0 is never a contact (no previous frame to compare)."""
        actions = np.zeros((5, 7))
        actions[0, -1] = -1.0
        actions[1:, -1] = 1.0
        contacts = detect_contact_timesteps(actions, min_gap=1)
        assert not contacts[0]

    def test_zero_velocity_not_reversal(self):
        """Near-zero velocity should not trigger a reversal."""
        actions = np.zeros((10, 7))
        actions[:, :6] = 1e-8  # near-zero velocity throughout
        actions[:, -1] = 1.0
        contacts = detect_contact_timesteps(actions, min_gap=1)
        assert not contacts.any()


class TestFeatureDatasetSaveLoad:
    """Tests for FeatureDataset serialization."""

    def test_roundtrip(self):
        """Save and load should produce identical data."""
        N, D, H, A = 20, 64, 5, 7
        ds = FeatureDataset(
            hidden_states=np.random.randn(N, D).astype(np.float32),
            action_chunks=np.random.randn(N, H, A).astype(np.float32),
            future_hidden_states=np.random.randn(N, D).astype(np.float32),
            episode_ids=np.arange(N),
            task_ids=np.zeros(N, dtype=int),
            timesteps=np.arange(N),
            is_contact=np.random.choice([True, False], N),
            is_success=np.ones(N, dtype=bool),
            horizon=H,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            ds.save(f.name)
            loaded = FeatureDataset.load(f.name)

        np.testing.assert_array_equal(ds.hidden_states, loaded.hidden_states)
        np.testing.assert_array_equal(ds.action_chunks, loaded.action_chunks)
        np.testing.assert_array_equal(ds.future_hidden_states, loaded.future_hidden_states)
        np.testing.assert_array_equal(ds.episode_ids, loaded.episode_ids)
        np.testing.assert_array_equal(ds.is_contact, loaded.is_contact)
        assert ds.horizon == loaded.horizon
