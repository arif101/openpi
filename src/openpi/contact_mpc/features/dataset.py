"""Build (h_t, action_chunk, h_{t+H}) datasets from LeRobot demo data.

Loads a LeRobot dataset, runs the frozen Pi0.5 VLM over each observation
to extract hidden states, and pairs them with action chunks and future
hidden states for world model training.

Contact labels are derived from gripper state changes in the action
sequence (same heuristic as build_waypoint_dataset.py), not from
force/torque sensors (which LIBERO does not expose).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FeatureRecord:
    """A single timestep's extracted features."""

    episode_id: int
    task_id: int
    timestep: int
    hidden_state: np.ndarray  # [hidden_dim]
    action_chunk: np.ndarray  # [H, action_dim] or [action_dim] for single step
    is_contact: bool  # True if this timestep is a contact event
    is_success: bool  # True if the episode was successful


@dataclasses.dataclass
class FeatureDataset:
    """Collection of extracted feature records with metadata."""

    hidden_states: np.ndarray  # [N, hidden_dim]
    action_chunks: np.ndarray  # [N, H, action_dim]
    future_hidden_states: np.ndarray  # [N, hidden_dim] — h_{t+H}
    episode_ids: np.ndarray  # [N]
    task_ids: np.ndarray  # [N]
    timesteps: np.ndarray  # [N]
    is_contact: np.ndarray  # [N] bool
    is_success: np.ndarray  # [N] bool
    horizon: int  # H used for action chunks / future state offset

    def save(self, path: str) -> None:
        """Save dataset to a .npz file."""
        np.savez_compressed(
            path,
            hidden_states=self.hidden_states,
            action_chunks=self.action_chunks,
            future_hidden_states=self.future_hidden_states,
            episode_ids=self.episode_ids,
            task_ids=self.task_ids,
            timesteps=self.timesteps,
            is_contact=self.is_contact,
            is_success=self.is_success,
            horizon=np.array(self.horizon),
        )
        logger.info(f"Saved FeatureDataset ({len(self.hidden_states)} records) to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureDataset":
        """Load dataset from a .npz file."""
        data = np.load(path)
        return cls(
            hidden_states=data["hidden_states"],
            action_chunks=data["action_chunks"],
            future_hidden_states=data["future_hidden_states"],
            episode_ids=data["episode_ids"],
            task_ids=data["task_ids"],
            timesteps=data["timesteps"],
            is_contact=data["is_contact"],
            is_success=data["is_success"],
            horizon=int(data["horizon"]),
        )


def detect_contact_timesteps(actions: np.ndarray, min_gap: int = 2) -> np.ndarray:
    """Detect contact events from action sequences using gripper heuristic.

    Same logic as build_waypoint_dataset.py:extract_keyframe_indices.
    Contact = gripper state change (sign flip on last action dim) or
    velocity direction reversal (dot product < -0.1 on dims 0:6).

    Args:
        actions: Action sequence of shape [T, action_dim].
        min_gap: Minimum gap between contact events.

    Returns:
        Boolean array of shape [T] where True = contact event.
    """
    T = len(actions)
    is_contact = np.zeros(T, dtype=bool)
    last_contact = -min_gap  # allow first frame

    for i in range(1, T):
        if i - last_contact < min_gap:
            continue

        curr, prev = actions[i], actions[i - 1]

        # Gripper state change: sign flip on last dim
        gripper_changed = (curr[-1] > 0) != (prev[-1] > 0)

        # Velocity direction reversal on first 6 dims
        curr_vel, prev_vel = curr[:6], prev[:6]
        cn, pn = np.linalg.norm(curr_vel), np.linalg.norm(prev_vel)
        if cn > 1e-6 and pn > 1e-6:
            dot = np.dot(curr_vel / cn, prev_vel / pn)
            direction_changed = dot < -0.1
        else:
            direction_changed = False

        if gripper_changed or direction_changed:
            is_contact[i] = True
            last_contact = i

    return is_contact
