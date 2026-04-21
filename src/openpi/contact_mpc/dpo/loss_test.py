"""Tests for the flow-matching DPO loss math.

These tests exercise the analytical properties of the loss without any
neural-net forward passes. Correctness of the gradient wiring through
Pi0.5 is covered by the training-integration tests (separate file).
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.contact_mpc.dpo.loss import (
    dpo_logits,
    dpo_loss_from_fm,
    flow_matching_regression_loss,
    sample_flow_matching_noise,
)


class TestFlowMatchingRegressionLoss:
    def test_zero_when_prediction_matches_target(self):
        v = np.random.randn(4, 10, 7).astype(np.float32)
        loss = flow_matching_regression_loss(v, v)
        assert loss.shape == (4,)
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_positive_when_prediction_differs(self):
        v_pred = np.zeros((3, 5, 7))
        v_target = np.ones((3, 5, 7))
        loss = flow_matching_regression_loss(v_pred, v_target)
        np.testing.assert_allclose(loss, 1.0, atol=1e-6)

    def test_shape_preserves_batch(self):
        v_pred = np.random.randn(8, 10, 7)
        v_target = np.random.randn(8, 10, 7)
        loss = flow_matching_regression_loss(v_pred, v_target)
        assert loss.shape == (8,)


class TestDPOLogits:
    def test_positive_when_theta_prefers_winner(self):
        # θ has lower loss on winner, higher loss on loser → positive logit
        logits = dpo_logits(
            l_fm_theta_win=np.array([0.5]),
            l_fm_ref_win=np.array([1.0]),
            l_fm_theta_lose=np.array([1.5]),
            l_fm_ref_lose=np.array([1.0]),
            beta=1.0,
        )
        assert logits[0] > 0

    def test_negative_when_theta_prefers_loser(self):
        logits = dpo_logits(
            l_fm_theta_win=np.array([1.5]),
            l_fm_ref_win=np.array([1.0]),
            l_fm_theta_lose=np.array([0.5]),
            l_fm_ref_lose=np.array([1.0]),
            beta=1.0,
        )
        assert logits[0] < 0

    def test_zero_when_theta_matches_ref(self):
        shared = np.array([1.0, 2.0, 0.5])
        logits = dpo_logits(shared, shared, shared + 0.1, shared + 0.1, beta=1.0)
        np.testing.assert_allclose(logits, 0.0, atol=1e-6)

    def test_scales_with_beta(self):
        l1 = dpo_logits(np.array([0.5]), np.array([1.0]),
                        np.array([1.5]), np.array([1.0]), beta=1.0)[0]
        l2 = dpo_logits(np.array([0.5]), np.array([1.0]),
                        np.array([1.5]), np.array([1.0]), beta=2.0)[0]
        assert pytest.approx(l2, rel=1e-6) == 2.0 * l1


class TestDPOLossFromFM:
    def _good_batch(self):
        # θ preferred: winners have lower L_FM, losers have higher
        return dict(
            l_fm_theta_win=np.array([0.3, 0.4, 0.2]),
            l_fm_ref_win=np.array([0.8, 0.9, 0.7]),
            l_fm_theta_lose=np.array([1.3, 1.4, 1.5]),
            l_fm_ref_lose=np.array([0.8, 0.9, 0.7]),
        )

    def test_returns_expected_keys(self):
        out = dpo_loss_from_fm(**self._good_batch(), beta=0.1)
        assert set(out.keys()) >= {
            "loss", "logits", "accuracy",
            "implicit_reward_win", "implicit_reward_lose",
        }

    def test_loss_is_scalar(self):
        out = dpo_loss_from_fm(**self._good_batch(), beta=0.1)
        assert out["loss"].ndim == 0 or out["loss"].shape == ()

    def test_accuracy_is_1_when_all_prefer_winner(self):
        out = dpo_loss_from_fm(**self._good_batch(), beta=0.1)
        assert out["accuracy"] == pytest.approx(1.0)

    def test_accuracy_is_0_when_all_prefer_loser(self):
        bad = dict(
            l_fm_theta_win=np.array([1.3, 1.4]),
            l_fm_ref_win=np.array([0.5, 0.5]),
            l_fm_theta_lose=np.array([0.2, 0.3]),
            l_fm_ref_lose=np.array([0.5, 0.5]),
        )
        out = dpo_loss_from_fm(**bad, beta=1.0)
        assert out["accuracy"] == pytest.approx(0.0)

    def test_implicit_reward_positive_for_winners_under_good_theta(self):
        out = dpo_loss_from_fm(**self._good_batch(), beta=0.5)
        assert (out["implicit_reward_win"] > 0).all()
        assert (out["implicit_reward_lose"] < 0).all()

    def test_loss_lower_when_theta_is_better(self):
        out_good = dpo_loss_from_fm(**self._good_batch(), beta=1.0)
        # Flip winner/loser losses: θ is now worse than reference on winners
        bad = dict(
            l_fm_theta_win=np.array([1.3, 1.4, 1.5]),
            l_fm_ref_win=np.array([0.5, 0.5, 0.5]),
            l_fm_theta_lose=np.array([0.2, 0.3, 0.4]),
            l_fm_ref_lose=np.array([0.5, 0.5, 0.5]),
        )
        out_bad = dpo_loss_from_fm(**bad, beta=1.0)
        assert out_good["loss"] < out_bad["loss"]


class TestSampleFlowMatchingNoise:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        action_chunk = np.random.randn(4, 10, 7).astype(np.float32)
        t, eps, x_t = sample_flow_matching_noise(action_chunk, rng)
        assert t.shape == (4, 1, 1)
        assert eps.shape == (4, 10, 7)
        assert x_t.shape == (4, 10, 7)

    def test_t_in_unit_interval(self):
        rng = np.random.default_rng(0)
        action_chunk = np.random.randn(100, 10, 7).astype(np.float32)
        t, _, _ = sample_flow_matching_noise(action_chunk, rng)
        assert (t >= 0.0).all() and (t <= 1.0).all()

    def test_x_t_is_correct_interpolation(self):
        rng = np.random.default_rng(0)
        action_chunk = np.random.randn(2, 4, 7).astype(np.float32)
        t, eps, x_t = sample_flow_matching_noise(action_chunk, rng)
        expected = (1.0 - t) * action_chunk + t * eps
        np.testing.assert_allclose(x_t, expected, atol=1e-6)

    def test_different_seeds_give_different_noise(self):
        action_chunk = np.random.randn(4, 10, 7).astype(np.float32)
        _, eps1, _ = sample_flow_matching_noise(action_chunk, np.random.default_rng(0))
        _, eps2, _ = sample_flow_matching_noise(action_chunk, np.random.default_rng(1))
        assert not np.allclose(eps1, eps2)
