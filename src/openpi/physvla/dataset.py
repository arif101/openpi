"""Torch Dataset over PhysVLA per-step physics traces.

Each .npz file produced by run_reason_v3_mppi.py --log-physics-traces holds
one episode with arrays of length T:
  ee_pos[T,3], ee_quat[T,4], ee_wrench[T,6], gripper_qpos[T,2],
  action[T,7], qpos[T,nq], qvel[T,nv],
  object_pos[T,n_obj,3], object_quat[T,n_obj,4], object_wrench[T,n_obj,6],
  contacts[T] (object dtype),
  image[K,256,256,3] (sparse, K = T // stride), image_t[K]

We yield single-step pairs (frame_t, frame_{t+1}) so the dynamics model can
learn z_t → z_{t+1} under a known action. Image lookup is interpolated: for
frame t, use the most-recent saved image at or before t.
"""

from __future__ import annotations

import glob
import pathlib
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _build_index(npz_paths: list[pathlib.Path]) -> list[tuple[int, int]]:
    """Walk every NPZ once, return [(file_idx, step_t)] for every valid t.

    We skip the final step of each episode (no t+1 target available).
    """
    idx: list[tuple[int, int]] = []
    for fi, p in enumerate(npz_paths):
        with np.load(p, allow_pickle=True) as d:
            T = int(d["t"].shape[0])
        if T < 2:
            continue
        for t in range(T - 1):
            idx.append((fi, t))
    return idx


def _nearest_image(image_t: np.ndarray, target_t: int) -> int:
    """Return index into image_t array for the most-recent saved image at or before target_t."""
    # image_t is sorted ascending. argmax of (image_t <= target_t) gives last match.
    mask = image_t <= target_t
    if not mask.any():
        return 0
    return int(mask.sum() - 1)


def _normalize_image(img: np.ndarray, target_hw: int = 224) -> np.ndarray:
    """Convert HWC uint8 to CHW float32 in [0, 1], center-cropped to target_hw."""
    # Naive center-crop; assumes input is square or close to. LIBERO is 256x256.
    H, W = img.shape[:2]
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    cropped = img[y0:y0 + s, x0:x0 + s]
    if s != target_hw:
        # Simple resize via numpy: nearest-neighbor downsample. Use stride.
        # For 256 → 224 we just take a 224x224 center crop instead of true resize.
        cropped = cropped[(s - target_hw) // 2:(s + target_hw) // 2,
                          (s - target_hw) // 2:(s + target_hw) // 2]
    chw = cropped.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw


class PhysVLAFrameDataset(Dataset):
    """One (frame_t, frame_{t+1}) pair per __getitem__."""

    def __init__(
        self,
        traces_dir: str | pathlib.Path,
        *,
        image_size: int = 224,
        n_objects: int = 12,
    ):
        self.traces_dir = pathlib.Path(traces_dir)
        self.image_size = image_size
        self.n_objects = n_objects
        self.npz_paths = sorted(self.traces_dir.glob("*.npz"))
        if not self.npz_paths:
            raise FileNotFoundError(
                f"No .npz files in {traces_dir}. Did the physics-traces logger run?"
            )
        # Cache loaded files lazily — index has (file_idx, step_t).
        self.index = _build_index(self.npz_paths)
        self._cache: dict[int, dict[str, Any]] = {}
        print(f"PhysVLAFrameDataset: {len(self.npz_paths)} episodes, "
              f"{len(self.index)} (t, t+1) pairs.")

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, fi: int) -> dict[str, Any]:
        if fi in self._cache:
            return self._cache[fi]
        with np.load(self.npz_paths[fi], allow_pickle=True) as d:
            entry = {k: d[k] for k in d.files}
        self._cache[fi] = entry
        return entry

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, t = self.index[idx]
        d = self._load(fi)
        tp1 = t + 1

        # Images: nearest saved frame at or before t / tp1.
        img_idx_t = _nearest_image(d["image_t"], int(d["t"][t]))
        img_idx_tp1 = _nearest_image(d["image_t"], int(d["t"][tp1]))
        img_t = _normalize_image(d["image"][img_idx_t], self.image_size)
        img_tp1 = _normalize_image(d["image"][img_idx_tp1], self.image_size)

        # Proprio at t / tp1: full physical state, padded to fixed dims so the
        # DataLoader can batch across scenes with different object counts.
        #
        # Includes:
        #   robot_qpos (9)             : joint angles (Franka 7 + gripper 2)
        #   robot_qvel (9)             : joint velocities
        #   ee_pos (3) / ee_quat (4)   : FK convenience
        #   gripper_qpos (2)           : redundant with robot_qpos but standard
        #   object_pos (n_obj×3)       : per-object world position, padded to n_obj
        #   object_quat (n_obj×4)      : per-object orientation
        #   object_lin_vel (n_obj×3)   : derived from finite-difference of object_pos
        #
        # Object linear velocity is critical: from a single image the model
        # cannot infer how fast objects are moving, and dynamics prediction
        # without velocity is degenerate. We compute it via finite difference
        # on the trace (always available since we have t and t+1 in the same
        # episode for the dynamics pairs).
        ROBOT_NQ = 9
        ROBOT_NV = 9
        n_obj = self.n_objects  # fixed cap (default 12) across all tasks

        def _pad_obj_2d(arr2d, target_n, axis_dim):
            out = np.zeros((target_n, axis_dim), dtype=np.float32)
            n_in = min(arr2d.shape[0], target_n)
            out[:n_in] = arr2d[:n_in]
            return out

        obj_pos_t  = _pad_obj_2d(d["object_pos"][t],  n_obj, 3)
        obj_quat_t = _pad_obj_2d(d["object_quat"][t], n_obj, 4)
        obj_pos_tp1  = _pad_obj_2d(d["object_pos"][tp1],  n_obj, 3)
        obj_quat_tp1 = _pad_obj_2d(d["object_quat"][tp1], n_obj, 4)
        # Linear velocity via simple finite difference (per timestep, not per
        # second — the model just needs a velocity signal, not SI units).
        obj_lin_vel_t   = obj_pos_tp1 - obj_pos_t
        # For t+1's velocity we'd need t+2; fall back to t→t+1 estimate (same).
        # The dynamics loss only consumes proprio_t (current state) and an
        # encoder of proprio_tp1 for the self-consistency target; the latter
        # doesn't need velocity to be exactly right since the encoder learns
        # whatever helps prediction.
        obj_lin_vel_tp1 = obj_lin_vel_t

        proprio_t = np.concatenate([
            d["qpos"][t, :ROBOT_NQ].astype(np.float32),
            d["qvel"][t, :ROBOT_NV].astype(np.float32),
            d["ee_pos"][t].astype(np.float32),
            d["ee_quat"][t].astype(np.float32),
            d["gripper_qpos"][t].astype(np.float32),
            obj_pos_t.flatten(),
            obj_quat_t.flatten(),
            obj_lin_vel_t.flatten(),
        ]).astype(np.float32)
        proprio_tp1 = np.concatenate([
            d["qpos"][tp1, :ROBOT_NQ].astype(np.float32),
            d["qvel"][tp1, :ROBOT_NV].astype(np.float32),
            d["ee_pos"][tp1].astype(np.float32),
            d["ee_quat"][tp1].astype(np.float32),
            d["gripper_qpos"][tp1].astype(np.float32),
            obj_pos_tp1.flatten(),
            obj_quat_tp1.flatten(),
            obj_lin_vel_tp1.flatten(),
        ]).astype(np.float32)

        action = d["action"][t].astype(np.float32)

        # Aux targets at t+1 (not t). The dynamics output latent has to encode
        # future physics to satisfy these losses — pose at t is in the proprio
        # input so predicting it would be tautological.
        wrench_tp1 = d["ee_wrench"][tp1].astype(np.float32)

        # Pad per-object pose at t+1 to n_objects × 7
        obj_pos_tp1_arr = d["object_pos"][tp1]
        obj_quat_tp1_arr = d["object_quat"][tp1]
        n_in_file = obj_pos_tp1_arr.shape[0]
        pose_tp1 = np.zeros((self.n_objects, 7), dtype=np.float32)
        n = min(n_in_file, self.n_objects)
        pose_tp1[:n, :3] = obj_pos_tp1_arr[:n]
        pose_tp1[:n, 3:] = obj_quat_tp1_arr[:n]

        # Slip velocity at t+1: object velocity relative to EE motion. Uses
        # central difference around t+1 if available, else forward-diff t→t+1.
        if tp1 + 1 < int(d["t"].shape[0]):
            obj_v_tp1 = (d["object_pos"][tp1 + 1, 0] - d["object_pos"][tp1, 0])
            ee_v_tp1 = (d["ee_pos"][tp1 + 1] - d["ee_pos"][tp1])
        else:
            obj_v_tp1 = (d["object_pos"][tp1, 0] - d["object_pos"][t, 0])
            ee_v_tp1 = (d["ee_pos"][tp1] - d["ee_pos"][t])
        slip_tp1 = (obj_v_tp1 - ee_v_tp1).astype(np.float32)

        return {
            "image_t":    torch.from_numpy(img_t),
            "image_tp1":  torch.from_numpy(img_tp1),
            "proprio_t":  torch.from_numpy(proprio_t),
            "proprio_tp1":torch.from_numpy(proprio_tp1),
            "action_t":   torch.from_numpy(action),
            "wrench_tp1": torch.from_numpy(wrench_tp1),
            "pose_tp1":   torch.from_numpy(pose_tp1),
            "slip_tp1":   torch.from_numpy(slip_tp1),
        }
