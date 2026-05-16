"""Atomic forward-rollout-and-cost primitive for REASON-VLA v3.

Foundation piece: take (initial MuJoCo state, candidate action chunk),
forward-simulate in MuJoCo, return per-step trajectory and scalar cost
decomposed into named components.

This is the building block for both:
- MPPI sampling refinement (K candidates, pick best by cost)
- Gradient refinement (where cost components are differentiable)

Design decisions:
- Pure forward-sim in MuJoCo (CPU). Fast (~10us/step on CPU MuJoCo).
- Costs computed in numpy. PyTorch wrapping deferred to the refiner layer.
- Cost components are structurally separated so the refiner can pick which
  to differentiate, sample over, or use as hard filters.
- No assumptions about Pi0.5; the evaluator only cares about (state, action).

Cost components:
- joint_limit: sum of joint-limit violations across the trajectory
- collision_count: total contact-event count (gross safety filter)
- end_effector_distance: final-step gripper position vs target (if provided)
- anchor: ||action_chunk - prior_action||² (preserves Pi0.5's information)

The refiner layer (later) decides how to weight + combine these.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import mujoco
import numpy as np


@dataclasses.dataclass
class RolloutResult:
    """Per-step trajectory and diagnostics from a forward simulation."""

    qpos_traj: np.ndarray            # [H+1, nq] including initial state
    qvel_traj: np.ndarray            # [H+1, nv]
    ee_pose_traj: np.ndarray         # [H+1, 7] world-frame end-effector (xyz + quaternion wxyz)
    contact_counts: np.ndarray       # [H] number of active contacts per step
    penetration_total: np.ndarray    # [H] sum of negative contact distances per step (real collisions)
    robot_penetration: np.ndarray    # [H] same as above but only contacts involving the robot
    joint_limit_margin: np.ndarray   # [H+1, nq] signed distance to nearest limit
                                     #   positive = inside the limit
                                     #   negative = violation magnitude


@dataclasses.dataclass
class CostBreakdown:
    """Named cost components + total. All scalar."""

    joint_limit: float           # sum of joint-limit violations
    collision_penetration: float # sum of (-min(0, dist)) over robot-involved contacts
    end_effector: float          # final EE position vs target (or 0 if no target)
    anchor: float                # ||action - prior||²
    total: float                 # weighted sum (using weights from PhysicsEvaluator)
    # Diagnostics (not in weighted total but exposed for analysis)
    raw_contact_count: float     # total contacts (incl. resting); for sanity-check only


@dataclasses.dataclass
class CostWeights:
    """Weights for combining the cost components into a scalar."""

    joint_limit: float = 10.0
    collision_penetration: float = 100.0  # large since penetrations are in meters; 1mm = 0.1 cost
    end_effector: float = 1.0
    anchor: float = 0.1


class PhysicsEvaluator:
    """Forward-simulate action chunks in a MuJoCo model and compute costs.

    Usage:
        ev = PhysicsEvaluator.from_xml_path("path/to/scene.xml")
        roll = ev.rollout(init_qpos, init_qvel, action_chunk)
        cost = ev.cost(roll, action_chunk, prior_action, ee_target)
        # Or all in one:
        roll, cost = ev.evaluate(init_qpos, init_qvel, action_chunk, prior, target)

    Notes:
        - One ``MjData`` instance per evaluator. For parallel sampling, create
          multiple evaluators (one per worker), each with its own data buffer.
        - Forward simulation uses ``mj_step`` with the model's default integrator.
          For LIBERO scenes this is RK4 or Euler depending on the MJCF.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_body_name: str = "robot0_eef",
        weights: Optional[CostWeights] = None,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.weights = weights or CostWeights()

        # End-effector body: look up by name; fall back to last body if missing
        self.ee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )
        if self.ee_body_id < 0:
            # Try a few common alternative names from LIBERO / Robosuite Franka
            for alt in ("gripper0_eef", "robot0_right_hand", "gripper0_grip_site",
                        "gripper", "hand", "eef"):
                self.ee_body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, alt
                )
                if self.ee_body_id >= 0:
                    break
        if self.ee_body_id < 0:
            self.ee_body_id = model.nbody - 1

        # Cache joint-limit info. Free / ball joints have jnt_limited == False.
        self._jnt_low = model.jnt_range[:, 0].copy()
        self._jnt_high = model.jnt_range[:, 1].copy()
        self._jnt_limited = model.jnt_limited.astype(bool).copy()
        # Per-joint qpos start index
        self._jnt_qposadr = model.jnt_qposadr.copy()
        self._jnt_type = model.jnt_type.copy()

        # Mark which geoms belong to the robot (for filtering contacts to
        # "robot vs world" rather than "table vs floor"). A geom is robot if
        # its body's ancestor chain includes a body whose name starts with
        # any of the known robot prefixes used by LIBERO / Robosuite.
        robot_body_prefixes = ("robot0_", "gripper0_", "panda")
        is_robot_body = np.zeros(model.nbody, dtype=bool)
        for b in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if any(name.startswith(p) for p in robot_body_prefixes):
                is_robot_body[b] = True
        # Propagate to descendants (a body is robot if any ancestor is robot)
        for b in range(model.nbody):
            parent = b
            while parent > 0:
                if is_robot_body[parent]:
                    is_robot_body[b] = True
                    break
                parent = model.body_parentid[parent]
        self._is_robot_geom = np.zeros(model.ngeom, dtype=bool)
        for g in range(model.ngeom):
            self._is_robot_geom[g] = is_robot_body[model.geom_bodyid[g]]

    @classmethod
    def from_xml_path(cls, xml_path: str, **kwargs) -> "PhysicsEvaluator":
        return cls(mujoco.MjModel.from_xml_path(xml_path), **kwargs)

    @classmethod
    def from_xml_string(cls, xml: str, **kwargs) -> "PhysicsEvaluator":
        return cls(mujoco.MjModel.from_xml_string(xml), **kwargs)

    def rollout(
        self,
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        action_chunk: np.ndarray,
    ) -> RolloutResult:
        """Forward-simulate action chunk from initial state.

        Args:
            init_qpos: [nq] generalized position
            init_qvel: [nv] generalized velocity
            action_chunk: [H, action_dim] sequence of control inputs to apply
                          one per ``mj_step``

        Returns:
            ``RolloutResult`` with per-step trajectory and diagnostics.
        """
        if init_qpos.shape[0] != self.model.nq:
            raise ValueError(
                f"init_qpos shape mismatch: got {init_qpos.shape}, "
                f"model.nq = {self.model.nq}"
            )
        if init_qvel.shape[0] != self.model.nv:
            raise ValueError(
                f"init_qvel shape mismatch: got {init_qvel.shape}, "
                f"model.nv = {self.model.nv}"
            )

        H, action_dim = action_chunk.shape
        if action_dim != self.model.nu:
            raise ValueError(
                f"action_chunk action_dim {action_dim} != model.nu {self.model.nu}"
            )

        d = self.data
        d.qpos[:] = init_qpos
        d.qvel[:] = init_qvel
        d.ctrl[:] = 0
        mujoco.mj_forward(self.model, d)

        qpos_traj = np.zeros((H + 1, self.model.nq), dtype=np.float64)
        qvel_traj = np.zeros((H + 1, self.model.nv), dtype=np.float64)
        ee_pose_traj = np.zeros((H + 1, 7), dtype=np.float64)
        contact_counts = np.zeros(H, dtype=np.int32)
        penetration_total = np.zeros(H, dtype=np.float64)
        robot_penetration = np.zeros(H, dtype=np.float64)
        joint_limit_margin = np.zeros((H + 1, self.model.njnt), dtype=np.float64)

        # Record initial state
        qpos_traj[0] = d.qpos
        qvel_traj[0] = d.qvel
        ee_pose_traj[0] = self._ee_pose(d)
        joint_limit_margin[0] = self._joint_limit_margin(d.qpos)

        for t in range(H):
            d.ctrl[:] = action_chunk[t]
            mujoco.mj_step(self.model, d)
            qpos_traj[t + 1] = d.qpos
            qvel_traj[t + 1] = d.qvel
            ee_pose_traj[t + 1] = self._ee_pose(d)
            contact_counts[t] = d.ncon
            pen_all, pen_robot = self._contact_penetration_stats(d)
            penetration_total[t] = pen_all
            robot_penetration[t] = pen_robot
            joint_limit_margin[t + 1] = self._joint_limit_margin(d.qpos)

        return RolloutResult(
            qpos_traj=qpos_traj,
            qvel_traj=qvel_traj,
            ee_pose_traj=ee_pose_traj,
            contact_counts=contact_counts,
            penetration_total=penetration_total,
            robot_penetration=robot_penetration,
            joint_limit_margin=joint_limit_margin,
        )

    def cost(
        self,
        rollout: RolloutResult,
        action_chunk: np.ndarray,
        prior_action: Optional[np.ndarray] = None,
        ee_target_xyz: Optional[np.ndarray] = None,
    ) -> CostBreakdown:
        """Compute named cost components + weighted total from a rollout.

        Args:
            rollout: result of ``self.rollout(...)``
            action_chunk: the action chunk that produced this rollout
            prior_action: Pi0.5's initial proposal (anchor target).
                          If None, anchor cost is 0.
            ee_target_xyz: [3] target end-effector position.
                          If None, end_effector cost is 0.

        Returns:
            ``CostBreakdown`` with named components and the weighted total.
        """
        # Joint-limit cost: sum of (-margin) where margin < 0, only on limited joints
        margin = rollout.joint_limit_margin  # [H+1, njnt]
        if self._jnt_limited.any():
            limited_margin = margin[:, self._jnt_limited]
            jl_cost = float(np.sum(np.maximum(0.0, -limited_margin)))
        else:
            jl_cost = 0.0

        # Collision-penetration cost: only counts ROBOT-involved contacts where
        # the contact distance is negative (real penetration). Stationary
        # resting contacts (table-on-floor, object-on-table) have dist ≈ 0 and
        # don't contribute. This is the bug-fix vs the v1 cost which counted
        # raw d.ncon (was dominated by ~100 baseline contacts per step).
        coll_cost = float(np.sum(rollout.robot_penetration))

        # End-effector cost: final-step EE position vs target
        if ee_target_xyz is not None:
            final_xyz = rollout.ee_pose_traj[-1, :3]
            ee_cost = float(np.linalg.norm(final_xyz - ee_target_xyz))
        else:
            ee_cost = 0.0

        # Anchor cost: ||action - prior||²
        if prior_action is not None:
            anchor_cost = float(np.sum((action_chunk - prior_action) ** 2))
        else:
            anchor_cost = 0.0

        w = self.weights
        total = (
            w.joint_limit * jl_cost
            + w.collision_penetration * coll_cost
            + w.end_effector * ee_cost
            + w.anchor * anchor_cost
        )

        return CostBreakdown(
            joint_limit=jl_cost,
            collision_penetration=coll_cost,
            end_effector=ee_cost,
            anchor=anchor_cost,
            total=total,
            raw_contact_count=float(np.sum(rollout.contact_counts)),
        )

    def evaluate(
        self,
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        action_chunk: np.ndarray,
        prior_action: Optional[np.ndarray] = None,
        ee_target_xyz: Optional[np.ndarray] = None,
    ) -> tuple[RolloutResult, CostBreakdown]:
        """Convenience: rollout + cost in one call."""
        roll = self.rollout(init_qpos, init_qvel, action_chunk)
        cost = self.cost(roll, action_chunk, prior_action, ee_target_xyz)
        return roll, cost

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _contact_penetration_stats(self, data: mujoco.MjData) -> tuple[float, float]:
        """Compute total and robot-only penetration depth at the current state.

        Returns:
            (penetration_total, robot_penetration) where each is the sum of
            ``max(0, -dist)`` over contacts (so resting contacts at dist≈0
            don't contribute; only real penetration does).
        """
        if data.ncon == 0:
            return 0.0, 0.0
        pen_all = 0.0
        pen_robot = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            depth = max(0.0, -float(c.dist))
            if depth == 0.0:
                continue
            pen_all += depth
            if self._is_robot_geom[c.geom1] or self._is_robot_geom[c.geom2]:
                pen_robot += depth
        return pen_all, pen_robot

    def _ee_pose(self, data: mujoco.MjData) -> np.ndarray:
        """Extract end-effector pose: (x, y, z, qw, qx, qy, qz)."""
        pos = data.xpos[self.ee_body_id]
        quat = data.xquat[self.ee_body_id]
        return np.concatenate([pos, quat])

    def _joint_limit_margin(self, qpos: np.ndarray) -> np.ndarray:
        """Signed distance to nearest joint limit per joint.

        Positive: inside the limit. Negative: violation magnitude.
        For unlimited joints (free, ball, slide without limits), returns +inf.
        """
        margin = np.full(self.model.njnt, np.inf, dtype=np.float64)
        for j in range(self.model.njnt):
            if not self._jnt_limited[j]:
                continue
            jt = self._jnt_type[j]
            # Limited HINGE / SLIDE joints have a single qpos value at jnt_qposadr[j]
            if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                q = qpos[self._jnt_qposadr[j]]
                margin[j] = min(q - self._jnt_low[j], self._jnt_high[j] - q)
            # BALL / FREE limits skipped — not directly comparable to a scalar range
        return margin
