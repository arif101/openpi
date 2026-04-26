"""Monte Carlo Tree Search over a latent world model.

AlphaZero-style PUCT search with:
  - Policy prior from the VLA (Pi0.5 action-chunk sampler)
  - Transition model: LatentWorldModel (hidden_state, action_chunk) -> next_hidden_state
  - Leaf evaluation: PairwiseValueFunction / PRM

Intentionally framework-agnostic for the tree logic — the WM, value fn,
and policy sampler are all injected callables. This lets us unit-test
the tree logic with tiny numpy stubs, and swap in real Pi0.5 + JAX
infrastructure at deployment.

Reference: VLA-Reasoner (arXiv 2509.22643), VLAPS (2508.12211),
WorldPlanner (2511.03077), AlphaZero (Silver et al. 2017).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Protocol

import numpy as np


class PolicySampler(Protocol):
    """Callable returning (K action chunks, K prior probabilities) given a latent state."""

    def __call__(
        self,
        hidden_state: np.ndarray,  # [hidden_dim]
        k: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:  # ([K, H, action_dim], [K])
        ...


class WorldModelFn(Protocol):
    """Callable predicting next hidden state from (hidden_state, action_chunk)."""

    def __call__(
        self,
        hidden_state: np.ndarray,  # [hidden_dim]
        action_chunk: np.ndarray,  # [H, action_dim]
    ) -> np.ndarray:  # [hidden_dim]
        ...


class ValueFn(Protocol):
    """Callable scoring a hidden state. Higher is better.

    Extra context (action_chunk leading to the leaf, the root frame, the task
    instruction) is optional — pure-latent value functions ignore it. Frame +
    task are needed by foundation PRMs (Robometer, Sonnet) that score over
    pixels and language rather than latents.
    """

    def __call__(
        self,
        hidden_state: np.ndarray,             # [hidden_dim]
        action_chunk: np.ndarray | None = None,  # edge that produced this leaf
        *,
        frame: np.ndarray | None = None,      # root-call image (HxWx3 uint8)
        task: str | None = None,              # natural-language instruction
    ) -> float:
        ...


@dataclasses.dataclass
class MCTSConfig:
    """Configuration for one MCTS decision."""

    num_simulations: int = 64       # tree simulations per decision
    width_k: int = 4                # candidate actions sampled per expansion
    max_depth: int = 4              # tree depth limit (accumulating WM error above this)
    c_puct: float = 1.4             # PUCT exploration constant
    discount: float = 1.0           # reward discount along rollout depth; 1.0 = no discount
    prior_temperature: float = 1.0  # softens / sharpens the policy prior


class MCTSNode:
    """One node in the search tree.

    Stores the latent state, visit statistics, policy prior (for the edge
    that led here from its parent), and children. Lazy-expanded: children
    are populated only when visited during simulation.
    """

    __slots__ = (
        "hidden_state", "parent", "action_chunk", "prior",
        "N", "W", "children", "depth", "is_expanded",
    )

    def __init__(
        self,
        hidden_state: np.ndarray | None,
        parent: "MCTSNode | None",
        action_chunk: np.ndarray | None,
        prior: float,
        depth: int,
    ):
        self.hidden_state = hidden_state
        self.parent = parent
        self.action_chunk = action_chunk  # edge from parent
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children: list[MCTSNode] = []
        self.depth = depth
        self.is_expanded = False

    @property
    def Q(self) -> float:
        """Average value seen for this node across all simulations."""
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, c_puct: float) -> float:
        """PUCT formula — exploration-augmented Q for child selection.

        Caller evaluates this on each child of a node under consideration;
        parent's N is accessed via self.parent.
        """
        if self.parent is None:
            return 0.0
        exploration = (
            c_puct * self.prior * math.sqrt(max(self.parent.N, 1)) / (1 + self.N)
        )
        return self.Q + exploration


class MCTSPlanner:
    """Monte Carlo Tree Search over a latent world model.

    Given a current observation's hidden state, returns the action chunk
    chosen by tree search. All three injectable callables (policy_sampler,
    world_model, value_fn) operate on numpy arrays so the planner itself
    is framework-agnostic.
    """

    def __init__(
        self,
        policy_sampler: PolicySampler,
        world_model: WorldModelFn,
        value_fn: ValueFn,
        config: MCTSConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.policy_sampler = policy_sampler
        self.world_model = world_model
        self.value_fn = value_fn
        self.config = config or MCTSConfig()
        self.rng = rng or np.random.default_rng(0)

    def plan(
        self,
        root_hidden_state: np.ndarray,
        *,
        frame: np.ndarray | None = None,
        task: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run MCTS and return (chosen_action_chunk, diagnostics).

        ``frame`` and ``task`` are forwarded to value_fn calls — needed by
        foundation PRMs that score (frame, task, action_chunk). Latent-only
        value functions ignore them.
        """
        root = MCTSNode(
            hidden_state=root_hidden_state,
            parent=None,
            action_chunk=None,
            prior=1.0,
            depth=0,
        )
        self._expand(root)

        for _ in range(self.config.num_simulations):
            leaf = self._select(root)
            if leaf.hidden_state is None:
                # Lazy transition: compute it now
                leaf.hidden_state = self.world_model(
                    leaf.parent.hidden_state, leaf.action_chunk,
                )
            if not leaf.is_expanded and leaf.depth < self.config.max_depth:
                self._expand(leaf)
            value = self.value_fn(
                leaf.hidden_state,
                leaf.action_chunk,
                frame=frame,
                task=task,
            )
            self._backprop(leaf, value)

        if not root.children:
            raise RuntimeError("MCTS root has no children — policy_sampler returned nothing?")

        # Choose the most-visited child (AlphaZero style)
        best = max(root.children, key=lambda c: c.N)

        diagnostics = {
            "root_visits": root.N,
            "child_visits": [c.N for c in root.children],
            "child_Q": [c.Q for c in root.children],
            "child_priors": [c.prior for c in root.children],
            "chosen_idx": root.children.index(best),
            "tree_size": self._count_nodes(root),
            "max_depth_reached": self._max_depth(root),
        }
        return best.action_chunk, diagnostics

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Walk down the tree using PUCT until we hit an unexpanded node."""
        node = root
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda c: c.puct_score(self.config.c_puct))
            if node.depth >= self.config.max_depth:
                break
        return node

    def _expand(self, node: MCTSNode) -> None:
        """Sample K candidate actions from the policy, create child nodes."""
        action_chunks, priors = self.policy_sampler(
            node.hidden_state,
            self.config.width_k,
            self.rng,
        )
        if len(action_chunks) != len(priors):
            raise ValueError(
                f"policy_sampler returned mismatched lengths: "
                f"{len(action_chunks)} chunks vs {len(priors)} priors"
            )
        # Normalize + temperature-sharpen the prior
        priors = np.asarray(priors, dtype=np.float64)
        if self.config.prior_temperature != 1.0:
            priors = priors ** (1.0 / self.config.prior_temperature)
        total = priors.sum()
        if total <= 0:
            # Degenerate prior — fall back to uniform
            priors = np.ones_like(priors) / len(priors)
        else:
            priors = priors / total

        for action, p in zip(action_chunks, priors):
            child = MCTSNode(
                hidden_state=None,          # compute lazily on first visit
                parent=node,
                action_chunk=np.asarray(action, dtype=np.float32),
                prior=float(p),
                depth=node.depth + 1,
            )
            node.children.append(child)
        node.is_expanded = True

    def _backprop(self, leaf: MCTSNode, leaf_value: float) -> None:
        """Walk up the tree, incrementing N and adding discounted value."""
        node: MCTSNode | None = leaf
        value = leaf_value
        while node is not None:
            node.N += 1
            node.W += value
            value *= self.config.discount
            node = node.parent

    @staticmethod
    def _count_nodes(root: MCTSNode) -> int:
        count = 0
        stack = [root]
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children)
        return count

    @staticmethod
    def _max_depth(root: MCTSNode) -> int:
        max_d = 0
        stack = [root]
        while stack:
            n = stack.pop()
            max_d = max(max_d, n.depth)
            stack.extend(n.children)
        return max_d
