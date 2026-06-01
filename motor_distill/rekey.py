"""Hindsight goal-relative re-keying for the keystone probe.

The keystone probe asks: is Pi0.5's motor manifold (HOW to move) separable from
its scene-memorization (WHAT to aim at)? We test it by distilling a head that
maps (goal-relative target, proprio) -> action chunk, with pixels+language
DROPPED. If that head still executes competent grasps when *told* the target,
the motor competence was separable.

This module is the data bridge: it reads a physics-trace NPZ produced by
`run_reason_v3_mppi.py --log-physics-traces` (baseline Pi0.5, MPPI off) and
hindsight-relabels each step with

    g = end-effector pose at t+H, expressed in the TARGET OBJECT's frame at t

i.e. "where the hand was headed, relative to the object" — a translation- and
gravity-rotation-relative target. Dropping the raw scene from the conditioning
is the memorization-stripping move; expressing the target in the object frame
is what makes pose-generalization available by construction.

CONVENTION GOTCHA (load-bearing): the trace mixes quaternion conventions.
  - object_quat comes from MuJoCo `body_xquat`  -> scalar-FIRST  [w, x, y, z]
  - ee_quat     comes from robosuite eef_quat    -> scalar-LAST   [x, y, z, w]
Everything here is normalized to scalar-first internally. test_rekey.py pins
this down (and the global-yaw invariance the equivariance claim depends on).
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


# ----------------------------------------------------------------------------
# quaternion helpers — all scalar-first [w, x, y, z] internally
# ----------------------------------------------------------------------------
def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """robosuite/scalar-last -> scalar-first."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by unit quaternion q (scalar-first)."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_yaw(q: np.ndarray) -> float:
    """Yaw (rotation about world +z / gravity axis) from a scalar-first quat."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(theta: float) -> np.ndarray:
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)], dtype=np.float64)


# ----------------------------------------------------------------------------
# goal-relative target: EE pose expressed in an object's frame
# ----------------------------------------------------------------------------
def ee_in_object_frame(ee_pos, ee_quat_wxyz, obj_pos, obj_quat_wxyz):
    """Return (rel_pos[3], rel_quat[4]) of the EE pose in the object frame.

    rel = T_obj^{-1} . T_ee :
        rel_pos  = R_obj^T (ee_pos - obj_pos)
        rel_quat = q_obj^{-1} * q_ee
    Invariant to any global SE(3) transform applied to BOTH frames (tested).
    """
    qobj = quat_normalize(obj_quat_wxyz)
    qee = quat_normalize(ee_quat_wxyz)
    rel_pos = quat_rotate(quat_conj(qobj), np.asarray(ee_pos, np.float64) - np.asarray(obj_pos, np.float64))
    rel_quat = quat_normalize(quat_mul(quat_conj(qobj), qee))
    return rel_pos, rel_quat


# ----------------------------------------------------------------------------
# target-object selection
# ----------------------------------------------------------------------------
def select_target_object(object_pos: np.ndarray, object_names, name: str | None = None) -> int:
    """Pick which object's frame defines the goal.

    If `name` is given, match it (substring, case-insensitive). Otherwise fall
    back to the most-displaced object over the episode (the manipulated one).
    """
    names = [str(n) for n in object_names]
    if name is not None:
        for i, nm in enumerate(names):
            if name.lower() in nm.lower():
                return i
        raise ValueError(f"target object {name!r} not in {names}")
    if object_pos.shape[1] == 0:
        raise ValueError("no scene objects in trace")
    disp = np.linalg.norm(object_pos[-1] - object_pos[0], axis=-1)  # [n_obj]
    return int(np.argmax(disp))


# ----------------------------------------------------------------------------
# dataset construction
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class RekeyConfig:
    horizon: int = 16          # H: action-chunk length / look-ahead for g
    target_name: str | None = None  # None -> most-displaced object


def build_pairs(npz_path: str | pathlib.Path, cfg: RekeyConfig = RekeyConfig()):
    """Read one physics-trace NPZ -> goal-relative distillation pairs.

    Returns a dict of stacked arrays (N = T - H usable steps):
        g_pos   [N, 3]      EE position at t+H in the target object's frame at t
        g_quat  [N, 4]      EE orientation at t+H in that frame (scalar-first)
        proprio [N, 9]      current ee_pos(3) + ee_quat_wxyz(4) + gripper(2)
        chunk   [N, H, A]   executed actions[t : t+H]
    Plus metadata: target_index, target_name, success, horizon.
    """
    d = np.load(npz_path, allow_pickle=True)
    ee_pos = d["ee_pos"].astype(np.float64)          # [T, 3]
    ee_quat_xyzw = d["ee_quat"].astype(np.float64)   # [T, 4] robosuite scalar-last
    grip = d["gripper_qpos"].astype(np.float64)      # [T, 2]
    actions = d["action"].astype(np.float64)         # [T, A]
    obj_pos = d["object_pos"].astype(np.float64)     # [T, n_obj, 3]
    obj_quat = d["object_quat"].astype(np.float64)   # [T, n_obj, 4] MuJoCo scalar-first
    names = list(d["object_names"])

    T = ee_pos.shape[0]
    H = cfg.horizon
    if T <= H:
        raise ValueError(f"trace too short: T={T} <= H={H}")

    ti = select_target_object(obj_pos, names, cfg.target_name)
    ee_quat_wxyz = np.stack([quat_xyzw_to_wxyz(q) for q in ee_quat_xyzw])  # [T, 4]

    g_pos, g_quat, proprio, chunk, otpos, otquat = [], [], [], [], [], []
    for t in range(T - H):
        rp, rq = ee_in_object_frame(
            ee_pos[t + H], ee_quat_wxyz[t + H], obj_pos[t, ti], obj_quat[t, ti])
        g_pos.append(rp)
        g_quat.append(rq)
        proprio.append(np.concatenate([ee_pos[t], ee_quat_wxyz[t], grip[t]]))
        chunk.append(actions[t:t + H])
        otpos.append(obj_pos[t, ti])          # target object world pose at t —
        otquat.append(obj_quat[t, ti])         # needed for yaw-canonicalization + closed-loop

    return {
        "g_pos": np.stack(g_pos),
        "g_quat": np.stack(g_quat),
        "proprio": np.stack(proprio),
        "chunk": np.stack(chunk),
        "obj_pos": np.stack(otpos),            # [N, 3]
        "obj_quat": np.stack(otquat),          # [N, 4] scalar-first
        "target_index": ti,
        "target_name": str(names[ti]),
        "success": bool(d["success"]),
        "horizon": H,
        "task_description": str(d["task_description"]) if "task_description" in d else "",
    }
