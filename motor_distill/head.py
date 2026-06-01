"""Distilled goal-relative motor head for the keystone probe.

A small flow-matching (rectified-flow) head that maps
    (goal-relative target g, proprio)  ->  action chunk [H, 7]
with NO pixels and NO language. Distilled from Pi0.5's executed chunks
(re-keyed by motor_distill/rekey.py). If this head can execute competent
grasps when *told* g, Pi0.5's motor manifold is separable from its
scene-memorization — the keystone.

Two variants, implemented as a SINGLE network with a canonicalization toggle so
the ablation is clean and equivariance is exact-by-construction:
  - plain        : object-relative inputs, world-yaw orientation. NOT equivariant.
  - equivariant  : additionally de-rotate inputs/outputs by the target object's
                   yaw (SO(2) about gravity). Output rotates with the scene by
                   construction -> pose-generalization for free.

The chunk action convention (LIBERO/robosuite OSC_POSE): [pos_delta(3, world),
axis_angle_delta(3, world), gripper(1)]. Canonicalization rotates the two
3-vectors about z; the gripper scalar is invariant.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# yaw (SO(2) about gravity) helpers — torch, batched. scalar-first quats.
# ----------------------------------------------------------------------------
def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """[...,4] scalar-first -> yaw angle [...]."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotz(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rotate 3-vectors v[...,3] about +z by ang[...]."""
    c, s = torch.cos(ang), torch.sin(ang)
    x, y, z = v.unbind(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)


def yaw_quat(ang: torch.Tensor) -> torch.Tensor:
    """ang[...] -> scalar-first quat [...,4] for rotation about +z."""
    z = torch.zeros_like(ang)
    return torch.stack([torch.cos(ang / 2), z, z, torch.sin(ang / 2)], dim=-1)


def quat_canon_sign(q: torch.Tensor) -> torch.Tensor:
    """Resolve the quaternion double-cover (q ~ -q): force scalar part >= 0, so
    the raw components are a unique function of the rotation. Without this, yaw
    wrapping past +-pi flips the canonical-orientation sign and breaks the
    conditioning's yaw-invariance."""
    return q * torch.where(q[..., 0:1] < 0, -1.0, 1.0)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ----------------------------------------------------------------------------
# canonicalization: map world-frame batch <-> object-yaw-canonical frame
# ----------------------------------------------------------------------------
def canon_psi(obj_quat: torch.Tensor, equivariant: bool) -> torch.Tensor:
    """Canonical yaw to remove. ψ=object-yaw if equivariant else 0."""
    return quat_yaw(obj_quat) if equivariant else torch.zeros(obj_quat.shape[:-1], device=obj_quat.device)


def cond_vector(g_pos, g_quat, proprio, obj_pos, obj_quat, equivariant: bool):
    """Build the [B,16] conditioning the net sees, in canonical frame.
    proprio = [ee_pos(3, world), ee_quat(4, world, wxyz), gripper(2)]."""
    psi = canon_psi(obj_quat, equivariant)                       # [B]
    ee_pos, ee_quat, grip = proprio[:, :3], proprio[:, 3:7], proprio[:, 7:9]
    ee_rel = rotz(ee_pos - obj_pos, -psi)                        # object-relative, de-yawed
    ee_quat_c = quat_canon_sign(quat_mul(yaw_quat(-psi), ee_quat))  # de-yawed, sign-resolved
    # g is already in the object frame (yaw-invariant) -> passes through unchanged
    return torch.cat([g_pos, quat_canon_sign(g_quat), ee_rel, ee_quat_c, grip], dim=-1), psi


def chunk_to_canonical(chunk, psi):
    """chunk[B,H,7] world -> canonical (rotate pos/rot-deltas by -ψ)."""
    pos, rot, grip = chunk[..., :3], chunk[..., 3:6], chunk[..., 6:7]
    p = psi[:, None].expand(-1, chunk.shape[1])
    return torch.cat([rotz(pos, -p), rotz(rot, -p), grip], dim=-1)


def chunk_from_canonical(chunk_c, psi):
    """canonical chunk -> world (rotate pos/rot-deltas by +ψ)."""
    pos, rot, grip = chunk_c[..., :3], chunk_c[..., 3:6], chunk_c[..., 6:7]
    p = psi[:, None].expand(-1, chunk_c.shape[1])
    return torch.cat([rotz(pos, p), rotz(rot, p), grip], dim=-1)


# ----------------------------------------------------------------------------
# flow-matching head
# ----------------------------------------------------------------------------
def time_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class MotorHead(nn.Module):
    def __init__(self, horizon=16, action_dim=7, cond_dim=16, hidden=256, t_dim=32):
        super().__init__()
        self.H, self.A, self.equivariant = horizon, action_dim, True
        self.chunk_dim = horizon * action_dim
        self.cond_enc = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(self.chunk_dim + hidden + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.chunk_dim))
        self.t_dim = t_dim

    def velocity(self, x_t, t, cond_emb):
        h = torch.cat([x_t, cond_emb, time_embed(t, self.t_dim)], dim=-1)
        return self.net(h)

    def loss(self, g_pos, g_quat, proprio, chunk, obj_pos, obj_quat):
        """Rectified-flow loss in canonical space."""
        cond, psi = cond_vector(g_pos, g_quat, proprio, obj_pos, obj_quat, self.equivariant)
        x1 = chunk_to_canonical(chunk, psi).reshape(chunk.shape[0], -1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * x1
        v_target = x1 - x0
        v_pred = self.velocity(x_t, t, self.cond_enc(cond))
        return ((v_pred - v_target) ** 2).mean()

    @torch.no_grad()
    def sample(self, g_pos, g_quat, proprio, obj_pos, obj_quat, steps=10):
        """Integrate the flow; return a WORLD-frame action chunk [B,H,7]."""
        cond, psi = cond_vector(g_pos, g_quat, proprio, obj_pos, obj_quat, self.equivariant)
        cond_emb = self.cond_enc(cond)
        B = g_pos.shape[0]
        x = torch.randn(B, self.chunk_dim, device=g_pos.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=g_pos.device)
            x = x + dt * self.velocity(x, t, cond_emb)
        chunk_c = x.reshape(B, self.H, self.A)
        return chunk_from_canonical(chunk_c, psi)
