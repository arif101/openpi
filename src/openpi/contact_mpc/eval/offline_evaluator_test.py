"""Tests for the offline LoRA candidate evaluator.

We use tiny world-model and value-function instances (randomly initialized)
so tests run fast and don't require any trained checkpoints. What we
verify is the scoring pipeline's correctness — shapes, padding/trimming,
aggregation, ranking — not the quality of scores themselves.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
import torch

from openpi.contact_mpc.eval.offline_evaluator import (
    CandidateScore,
    _pad_or_trim_chunk,
    aggregate_scores,
    rank_candidates,
    save_rankings,
    score_candidate_from_rollouts,
    score_hidden_action_pairs,
)
from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction
from openpi.contact_mpc.world_model.architecture import LatentWorldModel, WorldModelConfig


@pytest.fixture
def tiny_world_model() -> LatentWorldModel:
    config = WorldModelConfig(
        hidden_dim=16, action_dim=7, max_horizon=4,
        d_model=16, n_heads=2, n_layers=1, dropout=0.0,
    )
    torch.manual_seed(0)
    m = LatentWorldModel(config)
    m.eval()
    return m


@pytest.fixture
def tiny_value_fn() -> PairwiseValueFunction:
    torch.manual_seed(1)
    vf = PairwiseValueFunction(input_dim=16, hidden_dim=8)
    vf.eval()
    return vf


class TestPadOrTrim:
    def test_same_length_no_op(self):
        a = np.random.randn(5, 7).astype(np.float32)
        out = _pad_or_trim_chunk(a, 5)
        np.testing.assert_array_equal(out, a)

    def test_trims_when_longer(self):
        a = np.random.randn(10, 7).astype(np.float32)
        out = _pad_or_trim_chunk(a, 4)
        assert out.shape == (4, 7)
        np.testing.assert_array_equal(out, a[:4])

    def test_zero_pads_when_shorter(self):
        a = np.ones((2, 7), dtype=np.float32)
        out = _pad_or_trim_chunk(a, 5)
        assert out.shape == (5, 7)
        np.testing.assert_array_equal(out[:2], a)
        np.testing.assert_array_equal(out[2:], np.zeros((3, 7), dtype=np.float32))


class TestScoreHiddenActionPairs:
    def test_output_shape_matches_N(self, tiny_world_model, tiny_value_fn):
        N, H, D = 17, 4, 7
        hidden_dim = tiny_world_model.config.hidden_dim
        hs = np.random.randn(N, hidden_dim).astype(np.float32)
        ac = np.random.randn(N, H, D).astype(np.float32)
        scores = score_hidden_action_pairs(hs, ac, tiny_world_model, tiny_value_fn)
        assert scores.shape == (N,)
        assert scores.dtype == np.float32

    def test_handles_variable_horizon_via_padding(self, tiny_world_model, tiny_value_fn):
        hidden_dim = tiny_world_model.config.hidden_dim
        target_H = tiny_world_model.config.max_horizon
        hs = np.random.randn(3, hidden_dim).astype(np.float32)
        # action chunks with H < target_H
        ac_short = np.random.randn(3, target_H - 2, 7).astype(np.float32)
        scores = score_hidden_action_pairs(hs, ac_short, tiny_world_model, tiny_value_fn)
        assert scores.shape == (3,)

    def test_batch_invariant(self, tiny_world_model, tiny_value_fn):
        """Scores should not change when batch_size changes."""
        hidden_dim = tiny_world_model.config.hidden_dim
        H = tiny_world_model.config.max_horizon
        hs = np.random.RandomState(42).randn(10, hidden_dim).astype(np.float32)
        ac = np.random.RandomState(43).randn(10, H, 7).astype(np.float32)
        s1 = score_hidden_action_pairs(hs, ac, tiny_world_model, tiny_value_fn, batch_size=1)
        s2 = score_hidden_action_pairs(hs, ac, tiny_world_model, tiny_value_fn, batch_size=10)
        np.testing.assert_allclose(s1, s2, atol=1e-5)

    def test_raises_on_N_mismatch(self, tiny_world_model, tiny_value_fn):
        hs = np.random.randn(5, 16).astype(np.float32)
        ac = np.random.randn(3, 4, 7).astype(np.float32)  # mismatched N
        with pytest.raises(ValueError, match="N mismatch"):
            score_hidden_action_pairs(hs, ac, tiny_world_model, tiny_value_fn)


class TestAggregateScores:
    def test_aggregates_stats(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = aggregate_scores("cand_a", scores)
        assert out.num_observations == 5
        assert out.imagined_score_mean == pytest.approx(3.0)
        assert out.imagined_score_median == pytest.approx(3.0)
        assert out.imagined_score_std == pytest.approx(np.std(scores))
        assert len(out.per_observation_scores) == 5


class TestRankCandidates:
    def test_sorts_descending_by_mean(self):
        cands = [
            CandidateScore("c1", 3, 0.5, 0.1, 0.5, [0.4, 0.5, 0.6]),
            CandidateScore("c2", 3, 0.9, 0.1, 0.9, [0.8, 0.9, 1.0]),
            CandidateScore("c3", 3, 0.2, 0.1, 0.2, [0.1, 0.2, 0.3]),
        ]
        ranked = rank_candidates(cands)
        assert [c.candidate_name for c in ranked] == ["c2", "c1", "c3"]


class TestScoreCandidateFromRollouts:
    def _make_npz(self, path: pathlib.Path, N: int, hidden_dim: int, H: int, D: int = 7):
        np.savez(
            path,
            hidden_states=np.random.randn(N, hidden_dim).astype(np.float32),
            action_chunks=np.random.randn(N, H, D).astype(np.float32),
            episode_ids=np.arange(N) // 5,  # 5 decisions per episode
            task_ids=np.arange(N) // 10,
            timesteps=np.arange(N),
            is_success=np.zeros(N, dtype=bool),
        )

    def test_scores_all_when_no_held_out_filter(self, tmp_path, tiny_world_model, tiny_value_fn):
        p = tmp_path / "rollouts.npz"
        self._make_npz(p, N=20, hidden_dim=16, H=4)
        out = score_candidate_from_rollouts("cand_a", p, tiny_world_model, tiny_value_fn)
        assert out.num_observations == 20

    def test_held_out_filter_applies(self, tmp_path, tiny_world_model, tiny_value_fn):
        p = tmp_path / "rollouts.npz"
        self._make_npz(p, N=20, hidden_dim=16, H=4)
        # episode_ids are 0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3 (5 decisions each)
        out = score_candidate_from_rollouts(
            "cand_a", p, tiny_world_model, tiny_value_fn,
            held_out_episode_ids={1, 2},
        )
        assert out.num_observations == 10

    def test_held_out_filter_with_no_match_raises(self, tmp_path, tiny_world_model, tiny_value_fn):
        p = tmp_path / "rollouts.npz"
        self._make_npz(p, N=20, hidden_dim=16, H=4)
        with pytest.raises(ValueError, match="No decisions from held_out_episode_ids"):
            score_candidate_from_rollouts(
                "cand_a", p, tiny_world_model, tiny_value_fn,
                held_out_episode_ids={99, 100},
            )


class TestSaveRankings:
    def test_writes_sorted_json(self, tmp_path):
        cands = [
            CandidateScore("c1", 3, 0.5, 0.1, 0.5, [0.4, 0.5, 0.6]),
            CandidateScore("c2", 3, 0.9, 0.1, 0.9, [0.8, 0.9, 1.0]),
        ]
        out = tmp_path / "rankings.json"
        save_rankings(cands, out, extra={"world_model": "small_H10"})

        import json
        loaded = json.loads(out.read_text())
        assert loaded["num_candidates"] == 2
        assert loaded["ranked_candidates"][0]["candidate_name"] == "c2"
        assert loaded["world_model"] == "small_H10"
