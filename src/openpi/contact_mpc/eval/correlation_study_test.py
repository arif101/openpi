"""Tests for the imagined-vs-real correlation study."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from openpi.contact_mpc.eval.correlation_study import (
    classify_correlation_strength,
    compute_correlation,
    pair_by_candidate_and_task,
    save_correlation,
    save_scatter_plot,
)


class TestClassifyStrength:
    def test_strong_band(self):
        assert classify_correlation_strength(0.5) == "strong"
        assert classify_correlation_strength(0.8) == "strong"

    def test_medium_band(self):
        assert classify_correlation_strength(0.3) == "medium"
        assert classify_correlation_strength(0.45) == "medium"

    def test_weak_band(self):
        assert classify_correlation_strength(0.29) == "weak"
        assert classify_correlation_strength(-0.5) == "weak"
        assert classify_correlation_strength(0.0) == "weak"


class TestComputeCorrelation:
    def test_perfect_positive(self):
        x = np.arange(10, dtype=float)
        y = 2 * x + 1
        r = compute_correlation(x, y, n_bootstrap=100, seed=0)
        assert r.pearson_r == pytest.approx(1.0)
        assert r.strength == "strong"
        assert r.n_pairs == 10

    def test_perfect_negative(self):
        x = np.arange(10, dtype=float)
        y = -x
        r = compute_correlation(x, y, n_bootstrap=100, seed=0)
        assert r.pearson_r == pytest.approx(-1.0)
        assert r.strength == "weak"

    def test_uncorrelated(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        r = compute_correlation(x, y, n_bootstrap=200, seed=1)
        assert abs(r.pearson_r) < 0.2

    def test_ci_contains_true_r_for_correlated_data(self):
        rng = np.random.default_rng(0)
        n = 100
        x = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.5
        y = 0.8 * x + noise
        r = compute_correlation(x, y, n_bootstrap=500, seed=2)
        # With correlation ~0.8, CI should be well above 0
        assert r.ci_low > 0.3
        assert r.ci_high < 1.0 + 1e-9

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_raises_on_too_few_pairs(self):
        with pytest.raises(ValueError, match="at least 3 pairs"):
            compute_correlation(np.array([1.0, 2.0]), np.array([3.0, 4.0]))

    def test_degenerate_variance_returns_nan(self):
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        r = compute_correlation(x, y, n_bootstrap=50, seed=0)
        assert np.isnan(r.pearson_r)

    def test_labels_default_when_none(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        r = compute_correlation(x, y, n_bootstrap=50, seed=0)
        assert len(r.labels) == 3

    def test_custom_labels_preserved(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        r = compute_correlation(x, y, labels=["a", "b", "c"], n_bootstrap=50, seed=0)
        assert r.labels == ["a", "b", "c"]


class TestPairByCandidateAndTask:
    def test_aligns_common_pairs(self):
        imagined = {
            "A": {0: 0.1, 1: 0.2, 2: 0.3},
            "B": {0: 0.5, 1: 0.6},
        }
        real = {
            "A": {0: 0.7, 1: 0.8, 2: 0.9},
            "B": {0: 0.3, 1: 0.4, 2: 0.5},  # task 2 has no imagined score for B
        }
        i, r, labels = pair_by_candidate_and_task(imagined, real)
        assert i.shape == (5,)
        assert r.shape == (5,)
        assert labels[0] == "A@t0"
        assert "B@t2" not in labels  # filtered

    def test_returns_empty_when_no_intersection(self):
        i, r, labels = pair_by_candidate_and_task({"A": {0: 1.0}}, {"B": {0: 1.0}})
        assert i.shape == (0,)
        assert r.shape == (0,)
        assert labels == []


class TestSerialization:
    def test_save_correlation_writes_json(self, tmp_path: pathlib.Path):
        x = np.arange(10, dtype=float)
        y = 2 * x + 1
        r = compute_correlation(x, y, n_bootstrap=50, seed=0)
        out = tmp_path / "corr.json"
        save_correlation(r, out)
        loaded = json.loads(out.read_text())
        assert loaded["n_pairs"] == 10
        assert loaded["strength"] == "strong"
        assert loaded["pearson_r"] == pytest.approx(1.0)

    def test_save_scatter_plot_produces_png(self, tmp_path: pathlib.Path):
        # Skip if matplotlib not installed — test environment may vary.
        pytest.importorskip("matplotlib")
        x = np.arange(10, dtype=float) + np.random.RandomState(0).randn(10) * 0.1
        y = 2 * x + np.random.RandomState(1).randn(10) * 0.5
        r = compute_correlation(x, y, n_bootstrap=50, seed=0)
        out = tmp_path / "scatter.png"
        save_scatter_plot(r, out)
        assert out.exists()
        assert out.stat().st_size > 1000  # non-empty PNG
