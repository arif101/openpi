"""Neural Dynamic Policy (NDP) — the LEARNED, general version of the DMP.

A net (conditioned on physical/object state) outputs DMP forcing weights; a
DIFFERENTIABLE goal-attractor layer integrates them to a trajectory that
CONVERGES TO THE GOAL BY CONSTRUCTION. Trained end-to-end by behaviour cloning.

The generalization comes from the attractor STRUCTURE, not from memorized
appearance: a novel goal just works (the spring pulls there), so there is no
training-distribution edge to fall off. This is the committed-program bet —
attractor structure (validated by the hand-coded DMP) made learned + general.

Baseline `PlainHead`: same conditioning, but outputs the trajectory directly
(no attractor) — the memorizer we expect to fail on out-of-range goals.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DMPLayer(nn.Module):
    """Differentiable discrete DMP integration. forward(w, g, x0) -> [B, T, n_dim]
    trajectory converging to g by construction (forcing vanishes as phase->0)."""
    def __init__(self, n_dim=3, n_bf=20, T=24, az=25.0, a_s=4.0):
        super().__init__()
        self.n_dim, self.n_bf, self.T = n_dim, n_bf, T
        self.az, self.bz = az, az / 4.0
        t = np.linspace(0, 1, T)
        s = np.exp(-a_s * t)                                  # phase, 1->~0
        c = np.exp(-a_s * np.linspace(0, 1, n_bf))           # basis centers (phase)
        h = np.r_[1.0 / np.diff(c) ** 2, 1.0 / (c[-1] - c[-2]) ** 2] * 0.5
        psi = np.exp(-h[None] * (s[:, None] - c[None]) ** 2)  # [T, n_bf]
        forcing_basis = (psi / psi.sum(1, keepdims=True)) * s[:, None]  # [T, n_bf]
        self.register_buffer("forcing_basis", torch.tensor(forcing_basis, dtype=torch.float32))
        self.dt = float(t[1] - t[0])

    def forward(self, w, g, x0):
        # w [B, n_dim, n_bf], g/x0 [B, n_dim]
        f = torch.einsum("tk,bdk->btd", self.forcing_basis, w) * (g - x0)[:, None, :]  # [B,T,n_dim]
        x, v = x0, torch.zeros_like(x0)
        xs = []
        for t in range(self.T):
            vdot = self.az * (self.bz * (g - x) - v) + f[:, t, :]
            v = v + vdot * self.dt
            x = x + v * self.dt
            xs.append(x)
        return torch.stack(xs, dim=1)                         # [B, T, n_dim]


def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class NDP(nn.Module):
    """cond + goal -> forcing weights -> attractor trajectory to goal."""
    def __init__(self, cond_dim, n_dim=3, n_bf=20, T=24, hidden=256):
        super().__init__()
        self.n_dim, self.n_bf = n_dim, n_bf
        self.enc = _mlp(cond_dim, n_dim * n_bf, hidden)
        self.dmp = DMPLayer(n_dim, n_bf, T)

    def forward(self, cond, g, x0):
        w = self.enc(cond).reshape(cond.shape[0], self.n_dim, self.n_bf)
        return self.dmp(w, g, x0)                             # [B, T, n_dim]


class PlainHead(nn.Module):
    """cond + goal -> trajectory DIRECTLY (no attractor). The memorizer baseline."""
    def __init__(self, cond_dim, n_dim=3, T=24, hidden=256):
        super().__init__()
        self.n_dim, self.T = n_dim, T
        self.net = _mlp(cond_dim + 2 * n_dim, T * n_dim, hidden)

    def forward(self, cond, g, x0):
        h = torch.cat([cond, g, x0], dim=-1)
        return self.net(h).reshape(cond.shape[0], self.T, self.n_dim)
