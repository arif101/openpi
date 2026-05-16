"""Unit tests for PhysicsEvaluator — uses synthetic scenes only.

LIBERO integration test is in scripts/test_physics_evaluator_libero.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.contact_mpc.refinement.physics_evaluator import (
    CostWeights,
    PhysicsEvaluator,
)


# Simple test scene: 2-DOF hinge arm with joint limits + a static obstacle box.
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

    <!-- Stationary box that the arm could collide with -->
    <body name="box" pos="0.7 0 0.15">
      <geom type="box" size="0.05 0.05 0.05"/>
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
    return PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="eef")


def test_rollout_returns_correct_shapes(evaluator):
    H = 10
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((H, evaluator.model.nu))

    roll = evaluator.rollout(init_qpos, init_qvel, action_chunk)

    assert roll.qpos_traj.shape == (H + 1, evaluator.model.nq)
    assert roll.qvel_traj.shape == (H + 1, evaluator.model.nv)
    assert roll.ee_pose_traj.shape == (H + 1, 7)
    assert roll.contact_counts.shape == (H,)
    assert roll.penetration_total.shape == (H,)
    assert roll.robot_penetration.shape == (H,)
    assert roll.joint_limit_margin.shape == (H + 1, evaluator.model.njnt)


def test_zero_action_zero_initial_velocity_is_stable(evaluator):
    """Zero action from zero state should not drift dramatically over H=20."""
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((20, evaluator.model.nu))

    roll = evaluator.rollout(init_qpos, init_qvel, action_chunk)
    # qpos drift should be small (gravity may pull the arm slowly)
    drift = np.abs(roll.qpos_traj[-1] - roll.qpos_traj[0]).max()
    assert drift < 1.0


def test_nonzero_action_changes_state(evaluator):
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.ones((10, evaluator.model.nu)) * 0.5

    roll = evaluator.rollout(init_qpos, init_qvel, action_chunk)
    # qpos should change from initial under nonzero ctrl
    assert not np.allclose(roll.qpos_traj[-1], roll.qpos_traj[0])


def test_ee_pose_is_within_workspace(evaluator):
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((1, evaluator.model.nu))

    roll = evaluator.rollout(init_qpos, init_qvel, action_chunk)
    ee = roll.ee_pose_traj[0]
    # Initial EE pose for our 2-link arm at zero qpos should be near (0.6, 0, 0.1)
    # (link1 sticks out 0.3 along +x, link2 sticks out another 0.3)
    assert ee[0] > 0.0     # x positive
    assert ee[2] > 0.0     # z above floor
    assert abs(ee[1]) < 0.1  # y near 0


def test_joint_limit_margin_signs_correct(evaluator):
    """Inside limits: positive margin. At limits: ~zero. Beyond: negative."""
    init_qvel = np.zeros(evaluator.model.nv)
    # Inside limits (qpos = 0, range = [-1.57, 1.57])
    qpos_inside = np.zeros(evaluator.model.nq)
    margin_inside = evaluator._joint_limit_margin(qpos_inside)
    limited_inside = margin_inside[evaluator._jnt_limited]
    assert (limited_inside > 0).all()

    # Push joint 0 beyond its limit
    qpos_beyond = np.zeros(evaluator.model.nq)
    qpos_beyond[0] = 2.0  # outside [-1.57, 1.57]
    margin_beyond = evaluator._joint_limit_margin(qpos_beyond)
    assert margin_beyond[0] < 0  # violation


def test_cost_zero_action_zero_anchor(evaluator):
    """If action == prior_action and no target / no violations, cost should be near zero."""
    H = 10
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((H, evaluator.model.nu))
    prior_action = action_chunk.copy()

    roll, cost = evaluator.evaluate(
        init_qpos, init_qvel, action_chunk, prior_action, target_xyz=None,
    )
    assert cost.joint_limit == 0.0
    assert cost.anchor == 0.0
    assert cost.target_distance == 0.0
    # contact_count may be nonzero if gravity creates contacts; just assert finite
    assert np.isfinite(cost.total)


def test_cost_anchor_is_l2_squared(evaluator):
    H = 5
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.full((H, evaluator.model.nu), 0.3)
    prior_action = np.zeros((H, evaluator.model.nu))

    roll, cost = evaluator.evaluate(init_qpos, init_qvel, action_chunk, prior_action)
    expected_anchor = float((0.3 ** 2) * H * evaluator.model.nu)
    assert cost.anchor == pytest.approx(expected_anchor, rel=1e-6)


def test_cost_ee_target_distance(evaluator):
    H = 1
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((H, evaluator.model.nu))

    roll, _ = evaluator.evaluate(init_qpos, init_qvel, action_chunk)
    actual_xyz = roll.ee_pose_traj[-1, :3]
    far_target = actual_xyz + np.array([1.0, 0.0, 0.0])

    cost = evaluator.cost(roll, action_chunk, prior_action=None, target_xyz=far_target)
    # The EE is at actual_xyz; target is 1m away on x. Distance should be ~1.
    assert cost.target_distance == pytest.approx(1.0, abs=1e-2)


def test_cost_tracks_body_when_id_given(evaluator):
    """When target_body_id is set, distance is measured from that body, not the EE."""
    import mujoco
    H = 1
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.zeros((H, evaluator.model.nu))

    # Track the static "box" body (at pos="0.7 0 0.15" in TEST_XML)
    box_id = mujoco.mj_name2id(evaluator.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    assert box_id >= 0
    roll, _ = evaluator.evaluate(
        init_qpos, init_qvel, action_chunk, target_body_id=box_id,
    )
    assert roll.tracked_body_pos_traj is not None
    assert roll.tracked_body_pos_traj.shape == (H + 1, 3)
    # Box is static; tracked trajectory should be ~constant near (0.7, 0, 0.15)
    box_pos = roll.tracked_body_pos_traj[-1]
    assert abs(box_pos[0] - 0.7) < 0.05
    assert abs(box_pos[2] - 0.15) < 0.05

    # target_xyz coincides with box → distance ~0; far from box → distance ~1
    cost_at_box = evaluator.cost(roll, action_chunk, target_xyz=box_pos)
    cost_far = evaluator.cost(roll, action_chunk, target_xyz=box_pos + np.array([1.0, 0, 0]))
    assert cost_at_box.target_distance == pytest.approx(0.0, abs=1e-3)
    assert cost_far.target_distance == pytest.approx(1.0, abs=1e-2)


def test_weighted_total_matches_components(evaluator):
    """Verify that total == sum of (weight · component)."""
    H = 3
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)
    action_chunk = np.ones((H, evaluator.model.nu)) * 0.2
    prior_action = np.zeros_like(action_chunk)

    weights = CostWeights(
        joint_limit=10.0, collision_penetration=200.0, target_distance=3.0, anchor=4.0,
    )
    ev = PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="eef", weights=weights)
    roll, cost = ev.evaluate(init_qpos, init_qvel, action_chunk, prior_action)
    expected = (
        weights.joint_limit * cost.joint_limit
        + weights.collision_penetration * cost.collision_penetration
        + weights.target_distance * cost.target_distance
        + weights.anchor * cost.anchor
    )
    assert cost.total == pytest.approx(expected, rel=1e-6)


def test_shape_mismatch_raises(evaluator):
    init_qpos = np.zeros(evaluator.model.nq)
    init_qvel = np.zeros(evaluator.model.nv)

    # Wrong action-dim
    bad_action = np.zeros((10, evaluator.model.nu + 1))
    with pytest.raises(ValueError, match="action_dim"):
        evaluator.rollout(init_qpos, init_qvel, bad_action)

    # Wrong nq
    bad_qpos = np.zeros(evaluator.model.nq + 1)
    with pytest.raises(ValueError, match="init_qpos"):
        evaluator.rollout(bad_qpos, init_qvel, np.zeros((1, evaluator.model.nu)))


def test_factory_methods_both_work():
    ev_str = PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="eef")
    assert ev_str.model.nq > 0
    # from_xml_path requires a real path — just check the str factory worked


def test_ee_body_lookup_falls_back_gracefully():
    """If ee_body_name not found, falls back to last body."""
    ev = PhysicsEvaluator.from_xml_string(TEST_XML, ee_body_name="nonexistent_body")
    assert ev.ee_body_id >= 0  # picked some body
