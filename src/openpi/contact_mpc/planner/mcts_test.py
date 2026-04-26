"""Tests for the MCTS planner.

We inject tiny numpy stubs for the policy sampler, world model, and
value function, so tests are framework-agnostic and run in a few ms.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.contact_mpc.planner.mcts import (
    MCTSConfig,
    MCTSNode,
    MCTSPlanner,
)


def _make_policy_sampler(fixed_actions: np.ndarray, fixed_priors: np.ndarray):
    """Return a deterministic sampler that always yields the same K candidates."""

    def sampler(hidden_state, k, rng):
        assert k == len(fixed_actions)
        return fixed_actions, fixed_priors

    return sampler


def _make_biased_world_model():
    """world_model(h, a) = h + mean(a). Action shifts the latent predictably."""

    def wm(h, a):
        return h + float(np.mean(a))

    return wm


def _make_linear_value_fn():
    """value_fn(h) = sum(h). With biased WM, bigger actions -> bigger values."""

    def v(h, action_chunk=None, *, frame=None, task=None):
        return float(np.sum(np.atleast_1d(h)))

    return v


class TestMCTSNode:
    def test_Q_is_zero_for_unvisited(self):
        n = MCTSNode(hidden_state=None, parent=None, action_chunk=None, prior=1.0, depth=0)
        assert n.Q == 0.0

    def test_Q_averages_over_visits(self):
        n = MCTSNode(hidden_state=None, parent=None, action_chunk=None, prior=1.0, depth=0)
        n.N = 4
        n.W = 8.0
        assert n.Q == pytest.approx(2.0)

    def test_puct_root_returns_zero(self):
        root = MCTSNode(hidden_state=None, parent=None, action_chunk=None, prior=1.0, depth=0)
        assert root.puct_score(c_puct=1.4) == 0.0

    def test_puct_exploration_term_shrinks_with_visits(self):
        root = MCTSNode(hidden_state=None, parent=None, action_chunk=None, prior=1.0, depth=0)
        root.N = 100
        child = MCTSNode(hidden_state=None, parent=root, action_chunk=None, prior=0.5, depth=1)
        root.children.append(child)
        s_before = child.puct_score(c_puct=1.4)
        child.N = 50
        s_after = child.puct_score(c_puct=1.4)
        assert s_before > s_after


class TestMCTSPlanner:
    def _trivial_setup(self, k=4, hidden_dim=8):
        # Three action candidates: "small", "medium", "big".
        actions = np.stack([
            np.full((2, 7), -1.0),   # very negative mean
            np.zeros((2, 7)),         # zero mean
            np.full((2, 7), +1.0),   # positive mean — should win with value=sum(h)
            np.full((2, 7), +0.5),
        ]).astype(np.float32)[:k]
        priors = np.array([0.25, 0.25, 0.25, 0.25])[:k]
        return actions, priors, hidden_dim

    def test_plan_returns_valid_action_shape(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=20, width_k=4, max_depth=2),
        )
        root_h = np.zeros(hd, dtype=np.float32)
        action, diag = planner.plan(root_h)
        assert action.shape == (2, 7)

    def test_plan_prefers_positive_action_under_positive_value_fn(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=80, width_k=4, max_depth=2, c_puct=1.4),
        )
        root_h = np.zeros(hd, dtype=np.float32)
        _, diag = planner.plan(root_h)
        # The "big" positive action (index 2) should be most visited
        assert diag["chosen_idx"] in (2, 3)  # top two positive-mean actions

    def test_plan_respects_max_depth(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=100, width_k=4, max_depth=2),
        )
        _, diag = planner.plan(np.zeros(hd, dtype=np.float32))
        assert diag["max_depth_reached"] <= 2

    def test_all_children_visited_when_exploration_high(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=40, width_k=4, max_depth=1, c_puct=10.0),
        )
        _, diag = planner.plan(np.zeros(hd, dtype=np.float32))
        assert all(v > 0 for v in diag["child_visits"])

    def test_diagnostics_contain_expected_keys(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=20, width_k=4, max_depth=2),
        )
        _, diag = planner.plan(np.zeros(hd, dtype=np.float32))
        expected = {
            "root_visits", "child_visits", "child_Q", "child_priors",
            "chosen_idx", "tree_size", "max_depth_reached",
        }
        assert expected <= set(diag.keys())

    def test_visits_sum_equals_num_simulations(self):
        actions, priors, hd = self._trivial_setup()
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=30, width_k=4, max_depth=2),
        )
        _, diag = planner.plan(np.zeros(hd, dtype=np.float32))
        # Every simulation visits root exactly once
        assert diag["root_visits"] == 30
        # Children visits sum to at most num_simulations (one visit per sim
        # reaches exactly one top-level child)
        assert sum(diag["child_visits"]) == 30

    def test_raises_when_sampler_returns_nothing(self):
        def empty_sampler(h, k, rng):
            return np.zeros((0, 2, 7)), np.zeros((0,))

        planner = MCTSPlanner(
            policy_sampler=empty_sampler,
            world_model=_make_biased_world_model(),
            value_fn=_make_linear_value_fn(),
            config=MCTSConfig(num_simulations=10),
        )
        with pytest.raises(RuntimeError, match="no children"):
            planner.plan(np.zeros(8, dtype=np.float32))

    def test_prior_temperature_sharpens(self):
        # Make prior heavily skewed; temperature < 1 should sharpen further.
        actions = np.stack([np.zeros((2, 7)) for _ in range(4)]).astype(np.float32)
        priors = np.array([0.7, 0.15, 0.1, 0.05])
        planner = MCTSPlanner(
            policy_sampler=_make_policy_sampler(actions, priors),
            world_model=lambda h, a: h,  # identity WM
            value_fn=lambda h, a=None, *, frame=None, task=None: 0.0,  # flat value -> only prior matters
            config=MCTSConfig(
                num_simulations=20, width_k=4, max_depth=1,
                c_puct=1.4, prior_temperature=0.5,
            ),
        )
        _, diag = planner.plan(np.zeros(8, dtype=np.float32))
        # With flat values, visits should concentrate on the highest-prior child
        assert diag["chosen_idx"] == 0
