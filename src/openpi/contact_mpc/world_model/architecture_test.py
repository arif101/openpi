"""Tests for world model architecture."""

import torch
import numpy as np
import pytest

from openpi.contact_mpc.world_model.architecture import (
    LatentWorldModel,
    WorldModelConfig,
    SIZE_CONFIGS,
)


class TestLatentWorldModelShape:
    """Tests that the model produces correct output shapes."""

    @pytest.fixture
    def config(self):
        return WorldModelConfig(hidden_dim=64, action_dim=7, max_horizon=10, d_model=32, n_heads=2, n_layers=2)

    def test_output_shape(self, config):
        model = LatentWorldModel(config)
        h = torch.randn(4, 64)
        a = torch.randn(4, 10, 7)
        out = model(h, a)
        assert out.shape == (4, 64)

    def test_variable_horizon(self, config):
        """H=3 and H=10 should both work without shape mismatch."""
        model = LatentWorldModel(config)
        h = torch.randn(2, 64)
        out3 = model(h, torch.randn(2, 3, 7))
        out10 = model(h, torch.randn(2, 10, 7))
        assert out3.shape == (2, 64)
        assert out10.shape == (2, 64)

    def test_batch_size_one(self, config):
        model = LatentWorldModel(config)
        h = torch.randn(1, 64)
        a = torch.randn(1, 5, 7)
        out = model(h, a)
        assert out.shape == (1, 64)

    def test_no_nan_in_output(self, config):
        model = LatentWorldModel(config)
        h = torch.randn(4, 64)
        a = torch.randn(4, 10, 7)
        out = model(h, a)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))


class TestPermutationSensitivity:
    """The model must produce different outputs when action tokens are permuted.

    If this test fails, positional encoding is missing or broken, and the
    model is treating the action chunk as a bag of actions rather than a
    sequence. This is the test from the plan's architecture spec.
    """

    def test_permuted_actions_give_different_output(self):
        config = WorldModelConfig(hidden_dim=64, action_dim=7, max_horizon=10, d_model=32, n_heads=2, n_layers=2)
        model = LatentWorldModel(config)
        model.eval()

        torch.manual_seed(42)
        h = torch.randn(1, 64)
        a = torch.randn(1, 8, 7)

        # Shuffle the action tokens
        perm = torch.tensor([7, 3, 1, 5, 0, 6, 2, 4])
        a_shuffled = a[:, perm, :]

        with torch.no_grad():
            out_original = model(h, a)
            out_shuffled = model(h, a_shuffled)

        diff = torch.norm(out_original - out_shuffled).item()
        assert diff > 1e-6, (
            f"Permuted actions produced identical output (diff={diff}). "
            "Positional encoding is missing or broken."
        )


class TestGradientFlow:
    """Verify gradients flow from loss back through the model."""

    def test_gradient_flows_to_inputs(self):
        config = WorldModelConfig(hidden_dim=64, action_dim=7, max_horizon=10, d_model=32, n_heads=2, n_layers=2)
        model = LatentWorldModel(config)

        h = torch.randn(2, 64, requires_grad=True)
        a = torch.randn(2, 5, 7, requires_grad=True)
        target = torch.randn(2, 64)

        pred = model(h, a)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()

        assert h.grad is not None
        assert a.grad is not None
        assert torch.any(h.grad != 0), "Zero gradients on hidden state input"
        assert torch.any(a.grad != 0), "Zero gradients on action input"

    def test_all_params_have_gradients(self):
        config = WorldModelConfig(hidden_dim=64, action_dim=7, max_horizon=10, d_model=32, n_heads=2, n_layers=2)
        model = LatentWorldModel(config)

        h = torch.randn(2, 64)
        a = torch.randn(2, 5, 7)
        target = torch.randn(2, 64)

        pred = model(h, a)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.any(param.grad != 0), f"Zero gradient for {name}"


class TestSizeConfigs:
    """Verify the pre-defined size configs produce reasonable models."""

    @pytest.mark.parametrize("size_name", ["small", "medium", "large"])
    def test_config_builds(self, size_name):
        config = SIZE_CONFIGS[size_name]
        model = LatentWorldModel(config)
        h = torch.randn(1, config.hidden_dim)
        a = torch.randn(1, 5, config.action_dim)
        out = model(h, a)
        assert out.shape == (1, config.hidden_dim)

    def test_size_ordering(self):
        """Small < medium < large in parameter count."""
        counts = {name: LatentWorldModel(cfg).param_count() for name, cfg in SIZE_CONFIGS.items()}
        assert counts["small"] < counts["medium"] < counts["large"]

    def test_param_count_reasonable(self):
        """Check param counts are in expected ranges."""
        small = LatentWorldModel(SIZE_CONFIGS["small"]).param_count()
        medium = LatentWorldModel(SIZE_CONFIGS["medium"]).param_count()
        large = LatentWorldModel(SIZE_CONFIGS["large"]).param_count()
        assert 500_000 < small < 2_000_000, f"Small: {small:,}"
        assert 2_000_000 < medium < 10_000_000, f"Medium: {medium:,}"
        assert 10_000_000 < large < 40_000_000, f"Large: {large:,}"
