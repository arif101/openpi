"""General masked/windowed-depth node localizer for the relational scene graph (step 1a).

Turns an object's image location into a COARSE 3D world point from the object's OWN depth ->
works at ANY height (table, shelf, drawer, stack); it is NOT plane-bound, so it replaces the
tabletop-only `ray_plane` hack. The grounding head selects a node; this lifts that node to the
coarse relative goal the wrist servo then refines. Precision is the servo's job -> coarse is enough.

Depth convention (validated): robosuite `*_depth` must be row-FLIPPED to align with the camera
projection (the `depth[::-1]` fix). Without it, objects far from image center localize to z ~ -0.3
(the historical 19-60 cm bug). Use `flipped_depth()` to get the aligned metric map.

Deploy-clean: nothing here reads ground-truth poses. The center pixel / mask comes from the
detector at deploy; the sanity-check harness (diag_node_localizer.py) feeds an oracle pixel only
to isolate LOCALIZER accuracy from DETECTOR accuracy.
"""
from __future__ import annotations
import numpy as np


def flipped_depth(sim, obs_depth, cam="agentview"):
    """Linearized, row-aligned metric depth map (applies the validated flip fix)."""
    import robosuite.utils.camera_utils as cu
    return cu.get_real_depth_map(sim, np.asarray(obs_depth))[::-1].copy()


def _unproject(sim, rc, depth_real, R, cam):
    """rc = array of [row,col] pixels -> world points via the camera transform + metric depth."""
    import robosuite.utils.camera_utils as cu
    inv = np.linalg.inv(cu.get_camera_transform_matrix(sim, cam, R, R))
    dm = depth_real[None]
    return np.array([np.asarray(cu.transform_from_pixels_to_world(np.array([[r, c]]), dm, inv)[0])
                     for r, c in rc])


def localize(sim, center_px, depth_real, R, cam="agentview", win=12, z_mode="surface", step=1):
    """Coarse 3D world point of the object near `center_px` from a depth WINDOW.

    Object surface = upper-half WORLD-Z within the window (the object sits above its support,
    whatever the support's height) -> height-agnostic and rejects surrounding-surface bleed.
    z_mode='surface' returns the object top (what a top-down grasp targets); 'center' = full median.
    Returns float32[3] world point, or None if the window is empty.
    """
    r0 = int(min(max(center_px[0], 0), R - 1)); c0 = int(min(max(center_px[1], 0), R - 1))
    rc = [(r, c) for r in range(max(0, r0 - win), min(R, r0 + win + 1), step)
          for c in range(max(0, c0 - win), min(R, c0 + win + 1), step)]
    if not rc:
        return None
    pts = _unproject(sim, rc, depth_real, R, cam)
    xy = np.median(pts[:, :2], axis=0)
    if z_mode == "surface":
        zmed = np.median(pts[:, 2]); top = pts[pts[:, 2] >= zmed]
        z = float(np.median(top[:, 2])) if len(top) else float(np.median(pts[:, 2]))
    else:
        z = float(np.median(pts[:, 2]))
    return np.array([xy[0], xy[1], z], np.float32)


def localize_mask(sim, mask, depth_real, R, cam="agentview", z_mode="surface", max_px=600):
    """Same as localize() but from a boolean object MASK (deploy path: detector gives a mask).
    Subsamples to max_px pixels for speed. Returns float32[3] or None."""
    rc = np.argwhere(mask)
    if len(rc) == 0:
        return None
    if len(rc) > max_px:
        rc = rc[np.random.default_rng(0).choice(len(rc), max_px, replace=False)]
    pts = _unproject(sim, [(int(r), int(c)) for r, c in rc], depth_real, R, cam)
    xy = np.median(pts[:, :2], axis=0)
    if z_mode == "surface":
        zmed = np.median(pts[:, 2]); top = pts[pts[:, 2] >= zmed]
        z = float(np.median(top[:, 2])) if len(top) else float(np.median(pts[:, 2]))
    else:
        z = float(np.median(pts[:, 2]))
    return np.array([xy[0], xy[1], z], np.float32)
