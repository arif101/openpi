"""Tests for linear probe and ranking accuracy."""

import numpy as np
import pytest

from openpi.contact_mpc.features.probe import (
    evaluate_ranking_accuracy,
    score_hidden_states,
    train_linear_probe,
)


class TestTrainLinearProbe:
    """Tests for the linear probe training."""

    def test_separable_data_high_accuracy(self):
        """Linearly separable data should achieve near-perfect accuracy."""
        rng = np.random.RandomState(42)
        N = 200
        D = 16
        # Class 0: centered at -2, class 1: centered at +2
        X = np.vstack([
            rng.randn(N, D) - 2.0,
            rng.randn(N, D) + 2.0,
        ]).astype(np.float32)
        y = np.array([0] * N + [1] * N)

        model, acc = train_linear_probe(X, y, n_folds=3)
        assert acc > 0.95, f"Expected >0.95 on separable data, got {acc}"
        assert model is not None

    def test_random_data_near_chance(self):
        """Random labels should produce ~50% accuracy."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 16).astype(np.float32)
        y = rng.choice([0, 1], 200)

        _, acc = train_linear_probe(X, y, n_folds=3)
        assert 0.3 < acc < 0.7, f"Expected near chance on random data, got {acc}"

    def test_rejects_single_class(self):
        """Should raise ValueError if only one class is present."""
        X = np.random.randn(50, 8).astype(np.float32)
        y = np.ones(50, dtype=int)
        with pytest.raises(ValueError, match="both positive and negative"):
            train_linear_probe(X, y)


class TestScoreHiddenStates:
    """Tests for the scoring function."""

    def test_output_shape(self):
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(50, 8) - 2, rng.randn(50, 8) + 2]).astype(np.float32)
        y_train = np.array([0] * 50 + [1] * 50)
        model, _ = train_linear_probe(X_train, y_train, n_folds=2)

        X_test = rng.randn(10, 8).astype(np.float32)
        scores = score_hidden_states(model, X_test)
        assert scores.shape == (10,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_separable_scores_ordered(self):
        """On separable data, positive examples should score higher."""
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(100, 8) - 3, rng.randn(100, 8) + 3]).astype(np.float32)
        y_train = np.array([0] * 100 + [1] * 100)
        model, _ = train_linear_probe(X_train, y_train, n_folds=2)

        neg_scores = score_hidden_states(model, rng.randn(20, 8).astype(np.float32) - 3)
        pos_scores = score_hidden_states(model, rng.randn(20, 8).astype(np.float32) + 3)
        assert np.mean(pos_scores) > np.mean(neg_scores)


class TestEvaluateRankingAccuracy:
    """Tests for pairwise ranking accuracy (KS3 proxy)."""

    def test_perfect_separation(self):
        """Perfect separation should give 100% ranking accuracy."""
        rng = np.random.RandomState(42)
        X_train = np.vstack([rng.randn(100, 8) - 3, rng.randn(100, 8) + 3]).astype(np.float32)
        y_train = np.array([0] * 100 + [1] * 100)
        model, _ = train_linear_probe(X_train, y_train, n_folds=2)

        success_h = rng.randn(10, 8).astype(np.float32) + 3
        failure_h = rng.randn(10, 8).astype(np.float32) - 3
        rank_acc = evaluate_ranking_accuracy(model, success_h, failure_h)
        assert rank_acc > 0.95

    def test_random_near_chance(self):
        """Random features should give ~50% ranking accuracy."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 8).astype(np.float32)
        y_train = rng.choice([0, 1], 200)
        model, _ = train_linear_probe(X_train, y_train, n_folds=2)

        success_h = rng.randn(10, 8).astype(np.float32)
        failure_h = rng.randn(10, 8).astype(np.float32)
        rank_acc = evaluate_ranking_accuracy(model, success_h, failure_h)
        assert 0.2 < rank_acc < 0.8

    def test_returns_float_in_range(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 4) - 1, rng.randn(50, 4) + 1]).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50)
        model, _ = train_linear_probe(X, y, n_folds=2)

        rank_acc = evaluate_ranking_accuracy(
            model,
            rng.randn(5, 4).astype(np.float32) + 1,
            rng.randn(5, 4).astype(np.float32) - 1,
        )
        assert isinstance(rank_acc, float)
        assert 0.0 <= rank_acc <= 1.0
