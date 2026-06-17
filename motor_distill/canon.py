"""SE(2) canonicalization for the wrist motor (cheap, escnn-free test of the equivariance thesis). The motor's
position leak is the in-plane BEARING of the object (different bearing -> different arm config -> different wrist
viewpoint). We remove that DOF by construction: rotate world about gravity-z by phi = -atan2(goal_y, goal_x) so the
goal ALWAYS points to +x, run the motor in this canonical frame, then rotate predicted actions back by +theta. The
motor then only has to generalize over DISTANCE (1D), not bearing -> position-general in bearing BY CONSTRUCTION.
Applied identically to training data and at inference. (Eq.Bot-style canonicalization, SE(2); arXiv:2511.15194.)
"""
from __future__ import annotations
import numpy as np
from PIL import Image


def _R(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], np.float32)


def canon_angle(goal_rel) -> float:
    """theta = bearing of the (privileged/binder) relative goal in world XY."""
    return float(np.arctan2(float(goal_rel[1]), float(goal_rel[0])))


def rot_vec_xy(v, phi: float):
    """Rotate the XY of a 3-vector by phi (z unchanged). p_canon = R(phi) p."""
    out = np.asarray(v, np.float32).copy(); out[:2] = _R(phi) @ out[:2]; return out


def rot_chunk_xy(chunk, phi: float):
    """[T,7] action chunk: rotate position-xy (0,1) and orientation-xy (3,4) by phi; z/oz/grip unchanged.
    Row-vector convention: out_xy = xy @ R(phi)^T."""
    out = np.asarray(chunk, np.float32).copy(); Rt = _R(phi).T
    out[:, :2] = np.asarray(chunk, np.float32)[:, :2] @ Rt
    out[:, 3:5] = np.asarray(chunk, np.float32)[:, 3:5] @ Rt
    return out


def rot_image(img, phi: float):
    """Rotate wrist image content by phi (radians) about center. PIL rotates CCW by degrees."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).rotate(float(np.degrees(phi)), resample=Image.BILINEAR), np.uint8)


def canonicalize(wrist, goal_rel, proprio, chunk=None):
    """Map a sample into the goal-bearing-canonical frame (phi = -theta). Returns canon (wrist, goal_rel, proprio[,chunk])
    and theta so actions can be rotated back at inference. proprio = [ori_axisangle(3), gripper(2)] -> rotate ori xy."""
    theta = canon_angle(goal_rel); phi = -theta
    wc = rot_image(wrist, phi)
    gc = rot_vec_xy(goal_rel, phi)
    pc = np.asarray(proprio, np.float32).copy(); pc[:3] = rot_vec_xy(pc[:3], phi)   # rotate ee-orientation axis-angle xy
    if chunk is None:
        return wc, gc, pc, theta
    cc = rot_chunk_xy(chunk, phi)
    return wc, gc, pc, cc, theta


def decanon_chunk(chunk_canon, theta: float):
    """Rotate a predicted canonical action chunk back to the world frame (by +theta)."""
    return rot_chunk_xy(chunk_canon, theta)
