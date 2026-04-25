"""Tests for the world-model anti-collapse regularizers (InfoNCE, VICReg).

These loss terms are designed to attack the specific failure modes
diagnosed by scripts/diagnose_world_model.py:
  - InfoNCE: manifold drift (predicted features fall outside the real
    feature distribution)
  - VICReg: variance collapse (predicted features regress to conditional
    mean, losing per-dim spread and/or decorrelation)

These tests verify the losses have the mathematical properties that
justify their use, without running full training.
"""

from __future__ import annotations

import torch
import pytest

from openpi.contact_mpc.world_model.train import _infonce_loss, _vicreg_loss


class TestInfoNCELoss:
    def test_zero_when_predictions_are_exact(self):
        """If pred == target, InfoNCE = log(B) — the lower bound — not zero,
        but minimized when the diagonal dominates. Test: matched pairs
        give lower loss than random pairs."""
        torch.manual_seed(0)
        target = torch.randn(16, 32)
        matched = target.clone()
        random_pred = torch.randn(16, 32)
        L_matched = _infonce_loss(matched, target, temperature=0.1)
        L_random = _infonce_loss(random_pred, target, temperature=0.1)
        assert L_matched < L_random

    def test_gradient_points_toward_target(self):
        """Gradient of InfoNCE should push the prediction toward the real
        target (the diagonal positive), not away."""
        torch.manual_seed(0)
        target = torch.randn(8, 16)
        pred = torch.randn(8, 16, requires_grad=True)
        L = _infonce_loss(pred, target, temperature=0.1)
        L.backward()
        # After one gradient step, prediction should be closer to target in cosine sim
        with torch.no_grad():
            before_cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
            pred_new = pred - 0.1 * pred.grad
            after_cos = torch.nn.functional.cosine_similarity(pred_new, target, dim=-1).mean()
        assert after_cos > before_cos

    def test_decreases_with_temperature_lowering_when_matched(self):
        """With matched pairs, a lower temperature should give lower loss."""
        torch.manual_seed(0)
        x = torch.randn(32, 64)
        L_hot = _infonce_loss(x, x, temperature=1.0)
        L_cold = _infonce_loss(x, x, temperature=0.05)
        assert L_cold < L_hot

    def test_output_is_scalar(self):
        L = _infonce_loss(torch.randn(8, 16), torch.randn(8, 16), temperature=0.1)
        assert L.dim() == 0


class TestInfoNCEL2Variant:
    """L2-based InfoNCE — designed for feature distributions where cosine
    is degenerate (e.g., features cluster on a tight cone)."""

    def test_l2_matched_pairs_score_lower_than_random(self):
        torch.manual_seed(0)
        target = torch.randn(16, 32)
        random_pred = torch.randn(16, 32)
        L_matched = _infonce_loss(target, target, temperature=0.1, similarity="l2")
        L_random = _infonce_loss(random_pred, target, temperature=0.1, similarity="l2")
        assert L_matched < L_random

    def test_l2_works_when_cosine_is_degenerate(self):
        """Construct features where ALL pairs have cosine ~1.0 (cone-shaped
        distribution like Pi0.5's pooled hidden states). Cosine InfoNCE
        cannot discriminate; L2 InfoNCE can."""
        torch.manual_seed(0)
        # Build a "cone" of features: all share a strong common direction
        # plus a tiny per-sample perturbation in random directions.
        common_direction = torch.ones(64) * 10.0
        perturbations = torch.randn(16, 64) * 0.01
        targets = common_direction + perturbations
        # Cosine similarity between any two of these is ~1.0
        cos_sims = torch.nn.functional.cosine_similarity(
            targets.unsqueeze(0), targets.unsqueeze(1), dim=-1,
        )
        assert (cos_sims > 0.99).all().item(), "Setup failure: not enough collapse"

        # Cosine InfoNCE on matched pairs gives near-uniform logits (degenerate)
        L_cosine_matched = _infonce_loss(targets, targets, 0.1, similarity="cosine")
        # L2 InfoNCE can still discriminate via magnitude/distance
        L_l2_matched = _infonce_loss(targets, targets, 0.1, similarity="l2")

        # When cosine is degenerate, L_cosine_matched is near log(B); L2 can be much lower
        assert L_l2_matched < L_cosine_matched

    def test_l2_gradient_pulls_predictions_toward_targets(self):
        torch.manual_seed(0)
        target = torch.randn(8, 16)
        pred = torch.randn(8, 16, requires_grad=True)
        L = _infonce_loss(pred, target, temperature=0.1, similarity="l2")
        L.backward()
        with torch.no_grad():
            d_before = ((pred - target) ** 2).sum(dim=-1).mean()
            pred_new = pred - 0.05 * pred.grad
            d_after = ((pred_new - target) ** 2).sum(dim=-1).mean()
        assert d_after < d_before  # gradient step pulls predictions closer to targets

    def test_unknown_similarity_raises(self):
        with pytest.raises(ValueError, match="Unknown InfoNCE similarity mode"):
            _infonce_loss(torch.randn(4, 8), torch.randn(4, 8), 0.1, similarity="dot")


class TestVICRegLoss:
    def test_zero_when_output_has_unit_std_and_identity_cov(self):
        """Near-optimal input (per-dim std ≈ 1, dims ≈ decorrelated) should
        produce near-zero loss."""
        torch.manual_seed(0)
        # Large batch, small dim — per-dim std ≈ 1, cross-cov ≈ 0
        x = torch.randn(10000, 8)
        L = _vicreg_loss(x, target_std=1.0, variance_weight=1.0, covariance_weight=0.04)
        assert L.item() < 0.05

    def test_penalizes_collapse(self):
        """A collapsed prediction (all samples identical) should have very
        high variance-term loss."""
        collapsed = torch.zeros(32, 16)
        normal = torch.randn(32, 16)
        L_collapsed = _vicreg_loss(collapsed, target_std=1.0, variance_weight=1.0, covariance_weight=0.04)
        L_normal = _vicreg_loss(normal, target_std=1.0, variance_weight=1.0, covariance_weight=0.04)
        assert L_collapsed > L_normal
        assert L_collapsed.item() > 0.5  # variance term should be large

    def test_penalizes_perfectly_correlated_dims(self):
        """If all dimensions are identical copies of one random vector, the
        covariance term should be large."""
        torch.manual_seed(0)
        base = torch.randn(100, 1)
        correlated = base.repeat(1, 16)                  # [100, 16], all dims identical
        decorrelated = torch.randn(100, 16)
        L_corr = _vicreg_loss(correlated, target_std=1.0, variance_weight=0.0, covariance_weight=1.0)
        L_decorr = _vicreg_loss(decorrelated, target_std=1.0, variance_weight=0.0, covariance_weight=1.0)
        assert L_corr > L_decorr

    def test_gradient_pushes_toward_unit_variance(self):
        """Starting from low-variance output, the gradient should push each
        dim's variance upward."""
        torch.manual_seed(0)
        # Build a leaf tensor directly with low std, so .grad is populated
        x = (torch.randn(64, 8) * 0.1).detach().requires_grad_(True)
        L = _vicreg_loss(x, target_std=1.0, variance_weight=1.0, covariance_weight=0.0)
        L.backward()
        assert x.grad is not None
        with torch.no_grad():
            std_before = x.std(dim=0)
            x_new = x - 0.5 * x.grad
            std_after = x_new.std(dim=0)
        assert (std_after > std_before).all()

    def test_output_is_scalar(self):
        L = _vicreg_loss(torch.randn(16, 8))
        assert L.dim() == 0

    def test_per_dim_target_std_tensor(self):
        """When target_std is a [D] tensor, variance penalty applies per-dim."""
        torch.manual_seed(0)
        # Predictor output: 0.1 std on every dim — universal collapse
        x = (torch.randn(200, 8) * 0.1).detach()
        # Target: 0.5 std on every dim (larger than current)
        target = torch.full((8,), 0.5)
        L = _vicreg_loss(x, target_std=target, variance_weight=1.0, covariance_weight=0.0)
        assert L.item() > 0.3  # large variance penalty

    def test_per_dim_target_respects_individual_dims(self):
        """If target_std has a dim with 0, that dim shouldn't be penalized."""
        torch.manual_seed(0)
        x = (torch.randn(200, 4) * 0.01).detach()
        # Target: zero on 3 dims, high on 1 dim → penalty should be ~high/4
        target = torch.tensor([2.0, 0.0, 0.0, 0.0])
        L = _vicreg_loss(x, target_std=target, variance_weight=1.0, covariance_weight=0.0)
        # L = mean(max(0, target - std)) = mean([2.0 - ~0.01, 0 clipped, 0 clipped, 0 clipped])
        # ≈ mean([~2.0, 0, 0, 0]) = ~0.5
        assert 0.3 < L.item() < 0.7

    def test_per_dim_target_shape_mismatch_raises(self):
        x = torch.randn(16, 8)
        bad_target = torch.full((5,), 1.0)  # wrong dim
        with pytest.raises(ValueError, match="target_std tensor shape"):
            _vicreg_loss(x, target_std=bad_target)


class TestCombinedBackprop:
    """Sanity check that both regularizers can be combined with MSE and
    backpropagated through a small module without numerical issues."""

    def test_combined_loss_has_finite_gradient(self):
        torch.manual_seed(0)
        B, D = 16, 32
        pred = torch.randn(B, D, requires_grad=True)
        target = torch.randn(B, D)

        L_mse = torch.nn.functional.mse_loss(pred, target)
        L_nce = _infonce_loss(pred, target, temperature=0.1)
        L_vic = _vicreg_loss(pred, target_std=1.0,
                             variance_weight=1.0, covariance_weight=0.04)
        L = L_mse + 0.1 * L_nce + L_vic
        L.backward()

        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0
