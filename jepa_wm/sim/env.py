"""Planar pusher sim: a 2-DOF actuated pusher and a free ball on a table.

Action-conditioned dynamics with a predictable part (pusher kinematics) and a
less-predictable part (contact pushing the ball) — the predictable/unpredictable
split a JEPA is meant to exploit.

Anomaly hooks (for M2) let us inject physically-impossible events so we can test
whether the world-model's surprise signal spikes at them.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

_XML = pathlib.Path(__file__).with_name("pusher.xml")


class PusherEnv:
    def __init__(self, img_size: int = 64, frame_skip: int = 10, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(_XML))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, img_size, img_size)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        # qpos layout: [px, py, ball_x, ball_y, ball_z, ball_quat(4)]
        self._ball_qadr = self.model.jnt_qposadr[self.model.joint("ball_free").id]
        self._ball_vadr = self.model.jnt_dofadr[self.model.joint("ball_free").id]

    # ---- core API -------------------------------------------------------
    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        # randomize pusher and ball start within the arena
        self.data.qpos[0:2] = self.rng.uniform(-0.2, 0.2, size=2)
        bx, by = self.rng.uniform(-0.18, 0.18, size=2)
        self.data.qpos[self._ball_qadr + 0] = bx
        self.data.qpos[self._ball_qadr + 1] = by
        self.data.qpos[self._ball_qadr + 2] = 0.03
        self.data.qpos[self._ball_qadr + 3 : self._ball_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict:
        action = np.clip(np.asarray(action, dtype=np.float64), -0.3, 0.3)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._obs()

    # ---- anomaly injection (M2) ----------------------------------------
    # All anomalies are in-plane (observable from the top-down camera) and
    # violate the learned contact dynamics. Each returns a realized obs.
    def inject_teleport(self):
        """Ball instantly jumps to a new in-plane location — discontinuous, impossible."""
        self.data.qpos[self._ball_qadr + 0] = self.rng.uniform(-0.18, 0.18)
        self.data.qpos[self._ball_qadr + 1] = self.rng.uniform(-0.18, 0.18)
        self.data.qvel[self._ball_vadr : self._ball_vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step_ghost(self, action):
        """Ball accelerates across the table on its own while the pusher holds
        still — motion without contact, impossible."""
        hold = self.data.qpos[0:2].copy()  # freeze pusher (ignore action)
        theta = self.rng.uniform(0, 2 * np.pi)
        self.data.qvel[self._ball_vadr + 0] = 1.2 * np.cos(theta)
        self.data.qvel[self._ball_vadr + 1] = 1.2 * np.sin(theta)
        self.data.ctrl[:] = hold
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._obs()

    def displace_ball(self):
        """Jump the ball in-plane (call AFTER a normal step so the pusher has
        moved normally). Clean OBJECT anomaly: pusher behaved, ball did not."""
        self.data.qpos[self._ball_qadr + 0] = self.rng.uniform(-0.18, 0.18)
        self.data.qpos[self._ball_qadr + 1] = self.rng.uniform(-0.18, 0.18)
        self.data.qvel[self._ball_vadr : self._ball_vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step_pusher_stuck(self, action):
        """Pusher is commanded (action) but does NOT move; ball evolves normally.
        Clean EFFECTOR anomaly = faithful toy analog of execution-stuck
        (commanded effector motion, no realized motion)."""
        hold = self.data.qpos[0:2].copy()
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float64), -0.3, 0.3)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[0:2] = hold  # pin pusher
            self.data.qvel[0:2] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step_passthrough(self, action):
        """Pusher drives into the ball but the ball is pinned in place — contact
        ignored, impossible. Toy analog of execution-stuck (commanded, no effect)."""
        ball_xy = self.data.qpos[self._ball_qadr : self._ball_qadr + 2].copy()
        self.data.ctrl[:] = np.clip(np.asarray(action, dtype=np.float64), -0.3, 0.3)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[self._ball_qadr : self._ball_qadr + 2] = ball_xy  # pin
            self.data.qvel[self._ball_vadr : self._ball_vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    # ---- observation ----------------------------------------------------
    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="topdown")
        return self.renderer.render()  # (H, W, 3) uint8

    def _obs(self) -> dict:
        return {
            "image": self.render(),
            "proprio": np.array(
                [self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]],
                dtype=np.float32,
            ),
            "ball_pos": np.array(
                [self.data.qpos[self._ball_qadr + 0], self.data.qpos[self._ball_qadr + 1]],
                dtype=np.float32,
            ),
        }
