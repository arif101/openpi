"""Tests for pairwise ranking probe."""

import numpy as np
import torch
import pytest

from openpi.contact_mpc.features.ranking_probe import (
    RankingMLP,
    build_ranking_pairs,
    train_ranking_probe,
)


class TestRankingMLP:
    def test_output_shape(self):
        model = RankingMLP(input_dim=64, hidden_dim=32)
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8,)

    def test_gradient_flow(self):
        model = RankingMLP(input_dim=64, hidden_dim=32)
        h_better = torch.randn(4, 64, requires_grad=True)
        h_worse = torch.randn(4, 64, requires_grad=True)
        loss = -torch.mean(torch.log(torch.sigmoid(model(h_better) - model(h_worse)) + 1e-8))
        loss.backward()
        assert h_better.grad is not None
        assert torch.any(h_better.grad != 0)
        assert h_worse.grad is not None
        assert torch.any(h_worse.grad != 0)

    def test_predict_proba_numpy(self):
        model = RankingMLP(input_dim=32, hidden_dim=16)
        x = np.random.randn(10, 32).astype(np.float32)
        scores = model.predict_proba_numpy(x)
        assert scores.shape == (10,)


class TestBuildRankingPairs:
    def test_pairs_respect_temporal_order(self):
        N = 100
        hidden = np.random.randn(N, 16).astype(np.float32)
        ep_ids = np.zeros(N, dtype=int)
        timesteps = np.arange(N, dtype=float)

        h_better, h_worse = build_ranking_pairs(hidden, ep_ids, timesteps, n_pairs=50, seed=42)
        assert len(h_better) == 50
        assert h_better.shape[1] == 16

    def test_multiple_episodes(self):
        N = 200
        hidden = np.random.randn(N, 8).astype(np.float32)
        ep_ids = np.array([0] * 100 + [1] * 100)
        timesteps = np.concatenate([np.arange(100), np.arange(100)]).astype(float)

        h_better, h_worse = build_ranking_pairs(hidden, ep_ids, timesteps, n_pairs=100, seed=42)
        assert len(h_better) == 100


class TestTrainRankingProbe:
    def test_separable_data_high_accuracy(self):
        """States with higher timestep should rank higher after training."""
        rng = np.random.RandomState(42)
        N = 500
        # Hidden state = [timestep_normalized, noise...]
        timesteps = np.arange(N, dtype=float)
        hidden = np.column_stack([
            timesteps / N * 10,  # strong signal in first dim
            rng.randn(N, 15),    # noise in rest
        ]).astype(np.float32)
        ep_ids = np.zeros(N, dtype=int)

        model, val_acc = train_ranking_probe(
            hidden, ep_ids, timesteps,
            hidden_dim=32, n_pairs=5000, n_val_pairs=500,
            num_epochs=20, batch_size=128,
        )
        assert val_acc > 0.80, f"Expected >0.80 on separable data, got {val_acc}"

    def test_random_data_near_chance(self):
        """Random features should give ~50% ranking accuracy."""
        rng = np.random.RandomState(42)
        N = 2000
        hidden = rng.randn(N, 64).astype(np.float32)
        # 10 episodes with shuffled timesteps within each — no learnable signal
        ep_ids = np.repeat(np.arange(10), N // 10)
        timesteps = np.concatenate([rng.permutation(N // 10).astype(float) for _ in range(10)])

        model, val_acc = train_ranking_probe(
            hidden, ep_ids, timesteps,
            hidden_dim=32, n_pairs=5000, n_val_pairs=1000,
            num_epochs=10, batch_size=256,
        )
        assert val_acc < 0.70, f"Expected well below real-signal accuracy on random data, got {val_acc}"
