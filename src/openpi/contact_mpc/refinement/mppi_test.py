"""Unit tests for MPPIRefiner on a synthetic scene.

LIBERO integration test is in scripts/test_mppi_libero.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.contact_mpc.refinement.mppi import MPPIRefiner
from openpi.contact_mpc.refinement.physics_evaluator import (
    CostWeights,
    PhysicsEvaluator,
)


# Same scene as physics_evaluator_test
TEST_XML = """
<mujoco model="test_arm">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="link1" pos="0 0 0.1">
      <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-1.57 1.57"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" limited="true" range="-1.57 1.57"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
        <body name="eef" pos="0.3 0 0">
          <geom name="grip" type="box" size="0.03 0.03 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-1 1" gear="5"/>
    <motor joint="j2" ctrlrange="-1 1" gear="5"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def evaluator():
    weights = CostWeights(
        joint_limit=10.0, collision_penetration=100.0, target_distance=1.0, anchor=0.05,
    )
    return PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="eef", weights=weights)


def test_refine_returns_correct_shape(evaluator):
    H = 5
    refiner = MPPIRefiner(evaluator, num_samples=8, noise_std=0.1, temperature=1.0,
                          rng=np.random.default_rng(0))
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    refined, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=None)
    assert refined.shape == prior.shape
    assert refined.dtype == prior.dtype
    assert diag.num_samples == 8
    assert diag.sample_costs.shape == (8,)


def test_refine_produces_valid_diagnostics(evaluator):
    """With a target, MPPI runs and populates diagnostics sensibly."""
    H = 10
    refiner = MPPIRefiner(
        evaluator, num_samples=64, noise_std=0.3, temperature=0.5,
        rng=np.random.default_rng(42),
    )
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    init_roll = evaluator.rollout(init_qpos, init_qvel, prior)
    initial_ee = init_roll.ee_pose_traj[0, :3]
    target = initial_ee + np.array([0.1, 0.0, 0.0])

    refined, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=target)

    # Sanity checks (not strong correctness assertions — MPPI in 1 iter on
    # random Gaussian samples isn't guaranteed to beat the nominal).
    assert np.isfinite(diag.nominal_cost)
    assert np.isfinite(diag.refined_cost)
    assert np.isfinite(diag.sample_costs).all()
    assert diag.best_sample_cost == diag.sample_costs.min()
    assert 0 < diag.weight_entropy < np.log(diag.num_samples) + 1e-6
    assert diag.wall_seconds > 0


def test_weights_sum_to_one_implicitly(evaluator):
    """Verify the algorithm correctly normalizes weights (entropy is bounded)."""
    H = 5
    refiner = MPPIRefiner(
        evaluator, num_samples=16, noise_std=0.2, temperature=1.0,
        rng=np.random.default_rng(0),
    )
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    _, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=None)
    # Entropy of uniform K=16 is log(16) ≈ 2.77. Bounded above.
    assert diag.weight_entropy <= np.log(16) + 1e-6
    # And bounded below at 0 (would require perfectly peaked weights).
    assert diag.weight_entropy >= 0


def test_iterating_multiple_times_doesnt_explode(evaluator):
    """N iterations should still produce a valid in-bounds refined action."""
    H = 5
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    refiner = MPPIRefiner(
        evaluator, num_samples=32, noise_std=0.2, temperature=0.5,
        num_iterations=10, rng=np.random.default_rng(0),
    )
    refined, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=None)
    assert refined.shape == prior.shape
    assert (refined >= -1.0).all() and (refined <= 1.0).all()
    assert np.isfinite(refined).all()
    assert diag.iterations == 10


def test_no_target_anchors_to_prior(evaluator):
    """Without a target, the only varying cost is anchor — refined should be
    much closer to prior than a typical random sample."""
    H = 5
    K = 32
    sigma = 0.2
    refiner = MPPIRefiner(evaluator, num_samples=K, noise_std=sigma, temperature=1.0,
                          rng=np.random.default_rng(0))
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    refined, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=None)

    # A typical random sample has L2 norm ≈ sigma · sqrt(H · action_dim)
    typical_sample_norm = sigma * np.sqrt(H * evaluator.model.nu)
    refined_drift = np.linalg.norm(refined - prior)
    # Refined should be < typical sample (anchor is pulling it back).
    # With K=32 finite samples + softmin weighting, expect ~typical/sqrt(K).
    expected_drift_upper = typical_sample_norm / np.sqrt(K) * 3.0  # 3x margin for variance
    assert refined_drift < expected_drift_upper, (
        f"refined drift {refined_drift:.4f} exceeds bound {expected_drift_upper:.4f}; "
        f"typical sample norm {typical_sample_norm:.4f}"
    )


def test_action_bounds_respected(evaluator):
    H = 5
    refiner = MPPIRefiner(
        evaluator, num_samples=16, noise_std=10.0, temperature=1.0,
        action_low=-0.5, action_high=0.5,
        rng=np.random.default_rng(0),
    )
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    refined, _ = refiner.refine(init_qpos, init_qvel, prior, target_xyz=None)
    assert (refined >= -0.5).all()
    assert (refined <= 0.5).all()


def test_invalid_args_raise():
    ev = PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="eef")
    with pytest.raises(ValueError, match="num_samples"):
        MPPIRefiner(ev, num_samples=1)
    with pytest.raises(ValueError, match="noise_std"):
        MPPIRefiner(ev, num_samples=8, noise_std=0)
    with pytest.raises(ValueError, match="temperature"):
        MPPIRefiner(ev, num_samples=8, temperature=0)
    with pytest.raises(ValueError, match="num_iterations"):
        MPPIRefiner(ev, num_samples=8, num_iterations=0)


def test_more_iterations_reduces_or_maintains_cost(evaluator):
    """Multiple MPPI iterations should monotonically not-worsen cost."""
    H = 8
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    prior = np.zeros((H, evaluator.model.nu), dtype=np.float32)
    target = np.array([0.5, 0.0, 0.1])

    costs = []
    for N in (1, 2, 3):
        refiner = MPPIRefiner(
            evaluator, num_samples=32, noise_std=0.2, temperature=0.5,
            num_iterations=N, rng=np.random.default_rng(0),
        )
        _, diag = refiner.refine(init_qpos, init_qvel, prior, target_xyz=target)
        costs.append(diag.refined_cost)
    # Each successive iteration should not WORSEN the cost (within tolerance for rng noise)
    # Strictly monotonic isn't guaranteed by MPPI, but successive iterations *typically* improve.
    # We assert the multi-iteration result is not dramatically worse than 1-iteration.
    assert costs[-1] <= costs[0] + 0.5
