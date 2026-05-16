"""MPPI refinement on top of PhysicsEvaluator.

Algorithm (Model Predictive Path Integral control):
  1. Take a nominal action chunk a_0 (Pi0.5's prior).
  2. Sample K perturbations: a_k = a_0 + ε_k, ε_k ~ N(0, Σ).
  3. Forward-simulate each candidate in MuJoCo via PhysicsEvaluator.
  4. Compute cost J_k per candidate (physics + task + anchor).
  5. Softmin-weight: w_k ∝ exp(-J_k / λ).
  6. Refined action: a* = a_0 + Σ_k w_k · ε_k.

Optionally iterate N times (each iteration's a* becomes the new nominal).

The cost source is forward-simulated physics — NOT a learned VF. This is the
architectural lesson from Phase 1: learned scoring functions trained on
narrow data drift in adversarial directions when iteratively optimized
against. Forward-simulated physics doesn't drift.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np

from openpi.contact_mpc.refinement.physics_evaluator import (
    CostBreakdown,
    PhysicsEvaluator,
)


@dataclasses.dataclass
class MPPIDiagnostics:
    """Per-call diagnostics. Useful for debugging/tuning."""

    num_samples: int
    iterations: int
    nominal_cost: float          # cost of the input prior
    refined_cost: float          # cost of the output refined chunk
    sample_costs: np.ndarray     # [K] all sample costs in the final iteration
    best_sample_cost: float      # min over samples in the final iteration
    weight_entropy: float        # entropy of softmax weights (high = uniform, low = peaked)
    wall_seconds: float


class MPPIRefiner:
    """Refines a nominal action chunk via MPPI sampling.

    Usage:
        ev = PhysicsEvaluator.from_xml_string(xml, ee_body_name="robot0_eef")
        refiner = MPPIRefiner(ev, num_samples=64, noise_std=0.2, temperature=1.0)
        a_refined, diag = refiner.refine(qpos, qvel, prior_action, ee_target_xyz)
    """

    def __init__(
        self,
        evaluator: PhysicsEvaluator,
        num_samples: int = 64,
        noise_std: float = 0.2,
        temperature: float = 1.0,
        num_iterations: int = 1,
        action_low: float = -1.0,
        action_high: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        if num_samples < 2:
            raise ValueError(f"num_samples must be >= 2, got {num_samples}")
        if noise_std <= 0:
            raise ValueError(f"noise_std must be positive, got {noise_std}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
        self.evaluator = evaluator
        self.K = num_samples
        self.sigma = noise_std
        self.lam = temperature
        self.N = num_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.rng = rng or np.random.default_rng(0)

    def refine(
        self,
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        prior_action: np.ndarray,           # [H, action_dim]
        target_xyz: Optional[np.ndarray] = None,
        target_body_id: Optional[int] = None,
    ) -> tuple[np.ndarray, MPPIDiagnostics]:
        """Refine ``prior_action`` via MPPI sampling against physics cost.

        Args:
            target_xyz: where to drive the tracked body (or EE).
            target_body_id: MuJoCo body id to track; None → use end-effector.

        Returns:
            refined_action: [H, action_dim], same shape as prior_action
            diagnostics: ``MPPIDiagnostics`` with per-call telemetry
        """
        t0 = time.time()
        H, action_dim = prior_action.shape

        # Pre-compute nominal cost for diagnostics
        nominal_roll, nominal_cost_breakdown = self.evaluator.evaluate(
            init_qpos, init_qvel, prior_action,
            prior_action=prior_action,           # anchor cost is 0 by definition
            target_xyz=target_xyz,
            target_body_id=target_body_id,
        )
        nominal_cost = nominal_cost_breakdown.total

        nominal = prior_action.copy()
        sample_costs = np.zeros(self.K)
        for it in range(self.N):
            # Step 1: Sample K perturbations
            noise = self.rng.standard_normal((self.K, H, action_dim)) * self.sigma
            # Step 2-3: Construct K candidates (clipped to action bounds)
            candidates = np.clip(nominal[None, ...] + noise, self.action_low, self.action_high)

            # Step 4: Forward-simulate each and compute cost
            for k in range(self.K):
                _, cb = self.evaluator.evaluate(
                    init_qpos, init_qvel, candidates[k],
                    prior_action=prior_action,    # anchor always to ORIGINAL prior
                    target_xyz=target_xyz,
                    target_body_id=target_body_id,
                )
                sample_costs[k] = cb.total

            # Step 5: Softmin weights
            costs_shifted = sample_costs - sample_costs.min()
            weights = np.exp(-costs_shifted / self.lam)
            weights = weights / weights.sum()

            # Step 6: Weighted update of the nominal
            # weighted_perturbation = sum_k w_k · (candidate_k - nominal)
            weighted_perturbation = np.einsum(
                "k,khd->hd", weights, candidates - nominal[None, ...],
            )
            nominal = np.clip(nominal + weighted_perturbation, self.action_low, self.action_high)

        # Final cost of the refined action
        _, refined_cost_breakdown = self.evaluator.evaluate(
            init_qpos, init_qvel, nominal,
            prior_action=prior_action,
            target_xyz=target_xyz,
            target_body_id=target_body_id,
        )
        refined_cost = refined_cost_breakdown.total

        # Weight entropy from the final iteration's weights
        entropy = float(-np.sum(weights * np.log(weights + 1e-12)))

        diag = MPPIDiagnostics(
            num_samples=self.K,
            iterations=self.N,
            nominal_cost=float(nominal_cost),
            refined_cost=float(refined_cost),
            sample_costs=sample_costs.copy(),
            best_sample_cost=float(sample_costs.min()),
            weight_entropy=entropy,
            wall_seconds=time.time() - t0,
        )
        return nominal.astype(prior_action.dtype), diag
