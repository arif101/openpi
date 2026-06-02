"""Reach-with-obstacle env: pusher must reach a goal; an immovable peg blocks the
straight path. Pusher dynamics match pusher.xml exactly, so the self-model transfers.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

_XML = pathlib.Path(__file__).with_name("reach_obstacle.xml")


class ReachObstacleEnv:
    def __init__(self, img_size: int = 64, frame_skip: int = 10, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(_XML))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, img_size, img_size)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)

    def reset(self, start, jam=True):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = start
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -0.3, 0.3)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._obs()

    def render(self):
        self.renderer.update_scene(self.data, camera="topdown")
        return self.renderer.render()

    def _obs(self):
        return {
            "image": self.render(),
            "proprio": np.array(
                [self.data.qpos[0], self.data.qpos[1], self.data.qvel[0], self.data.qvel[1]], np.float32
            ),
        }
