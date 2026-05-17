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

        # Proprio at t / tp1: concat (qpos, qvel, ee_pos, ee_quat, gripper_qpos).
        proprio_t = np.concatenate([
            d["qpos"][t], d["qvel"][t],
            d["ee_pos"][t], d["ee_quat"][t],
            d["gripper_qpos"][t],
        ]).astype(np.float32)
        proprio_tp1 = np.concatenate([
            d["qpos"][tp1], d["qvel"][tp1],
            d["ee_pos"][tp1], d["ee_quat"][tp1],
            d["gripper_qpos"][tp1],
        ]).astype(np.float32)

        action = d["action"][t].astype(np.float32)
        wrench = d["ee_wrench"][t].astype(np.float32)

        # Pad/truncate per-object pose to n_objects × 7
        obj_pos = d["object_pos"][t]    # [n_obj_in_file, 3]
        obj_quat = d["object_quat"][t]
        n_in_file = obj_pos.shape[0]
        pose = np.zeros((self.n_objects, 7), dtype=np.float32)
        n = min(n_in_file, self.n_objects)
        pose[:n, :3] = obj_pos[:n]
        pose[:n, 3:] = obj_quat[:n]

        # Slip velocity: object linear velocity in EE frame.
        # Approximation: (object_pos[t+1] - object_pos[t]) / dt for the first
        # tracked object, then transformed to be relative to EE motion.
        # For PhysVLA v1 we just use world-frame relative velocity of obj_0.
        obj_v = (d["object_pos"][tp1, 0] - d["object_pos"][t, 0])
        ee_v = (d["ee_pos"][tp1] - d["ee_pos"][t])
        slip = (obj_v - ee_v).astype(np.float32)

        return {
            "image_t":    torch.from_numpy(img_t),
            "image_tp1":  torch.from_numpy(img_tp1),
            "proprio_t":  torch.from_numpy(proprio_t),
            "proprio_tp1":torch.from_numpy(proprio_tp1),
            "action_t":   torch.from_numpy(action),
            "wrench_t":   torch.from_numpy(wrench),
            "pose_t":     torch.from_numpy(pose),
            "slip_t":     torch.from_numpy(slip),
        }
