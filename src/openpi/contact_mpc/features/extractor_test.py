"""Tests for VLM feature extraction.

Tests shape correctness, determinism, and pooling behavior without
materializing full model weights (uses nnx.eval_shape where possible).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.contact_mpc.features.extractor import pool_with_mask


class TestPoolWithMask:
    """Tests for the mean-pooling helper."""

    def test_shape(self):
        hidden = jnp.ones((2, 5, 8))  # batch=2, seq=5, dim=8
        mask = jnp.ones((2, 5), dtype=jnp.bool_)
        result = pool_with_mask(hidden, mask)
        assert result.shape == (2, 8)

    def test_uniform_values(self):
        """All-ones input with full mask should pool to all-ones."""
        hidden = jnp.ones((1, 4, 3))
        mask = jnp.ones((1, 4), dtype=jnp.bool_)
        result = pool_with_mask(hidden, mask)
        np.testing.assert_allclose(result, np.ones((1, 3)), atol=1e-6)

    def test_mask_excludes_padding(self):
        """Masked-out tokens should not contribute to the pool."""
        hidden = jnp.array([[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]])  # [1, 3, 2]
        mask = jnp.array([[True, True, False]])  # only first two tokens valid
        result = pool_with_mask(hidden, mask)
        # Mean of [1, 2] and [10, 20] = [5.5, 11.0]
        np.testing.assert_allclose(result, np.array([[5.5, 11.0]]), atol=1e-6)

    def test_all_masked_returns_zero(self):
        """If all tokens are masked, should return zero (not NaN/inf)."""
        hidden = jnp.ones((1, 3, 4))
        mask = jnp.zeros((1, 3), dtype=jnp.bool_)
        result = pool_with_mask(hidden, mask)
        assert not jnp.any(jnp.isnan(result))
        assert not jnp.any(jnp.isinf(result))
        np.testing.assert_allclose(result, np.zeros((1, 4)), atol=1e-6)

    def test_deterministic(self):
        """Same input should always produce same output."""
        hidden = jnp.array(np.random.randn(3, 7, 16).astype(np.float32))
        mask = jnp.ones((3, 7), dtype=jnp.bool_)
        r1 = pool_with_mask(hidden, mask)
        r2 = pool_with_mask(hidden, mask)
        np.testing.assert_array_equal(r1, r2)

    def test_batch_independence(self):
        """Each batch element should be pooled independently."""
        h1 = jnp.ones((1, 4, 2)) * 2.0
        h2 = jnp.ones((1, 4, 2)) * 8.0
        hidden = jnp.concatenate([h1, h2], axis=0)
        mask = jnp.ones((2, 4), dtype=jnp.bool_)
        result = pool_with_mask(hidden, mask)
        np.testing.assert_allclose(result[0], np.array([2.0, 2.0]), atol=1e-6)
        np.testing.assert_allclose(result[1], np.array([8.0, 8.0]), atol=1e-6)


class TestExtractVlmFeaturesShape:
    """Shape-level test for Pi0.extract_vlm_features using eval_shape.

    This test verifies that the method produces the correct output shape
    without materializing model weights or running a real forward pass.
    Requires JAX but not a GPU.
    """

    def test_output_shape(self):
        """extract_vlm_features should return [batch, hidden_dim]."""
        import flax.nnx as nnx
        from openpi.models import pi0_config

        config = pi0_config.Pi0Config(pi05=True)

        # Use eval_shape to trace without materializing weights
        abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

        # Build a fake observation spec matching the model's expected input
        batch_size = 2
        obs_spec, _ = config.inputs_spec(batch_size=batch_size)

        # Trace extract_vlm_features to get output shape
        output_shape = jax.eval_shape(abstract_model.extract_vlm_features, obs_spec)

        assert output_shape.shape[0] == batch_size
        assert len(output_shape.shape) == 2  # [batch, hidden_dim]
        # hidden_dim should be positive
        assert output_shape.shape[1] > 0
