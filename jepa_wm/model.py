"""Tiny action-conditioned JEPA.

  online encoder f_theta : image -> z         (trained)
  target encoder f_xi    : image -> z'        (EMA of f_theta, stop-grad target)
  predictor      P       : (z_t, a_t) -> ẑ    (predicts target latent of next frame)

Anti-collapse: BYOL/JEPA-style EMA+stop-grad asymmetry, PLUS an explicit VICReg
variance/covariance regularizer on the online embeddings — the term LeCun flags as
the central engineering problem. We also *measure* embedding std and effective rank
as the M1 gate, so collapse can't hide.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim: int = 128, in_ch: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),    # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),   # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(), # 16->8
            nn.Conv2d(128, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),# 8->4
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):  # x: (B,3,64,64) in [0,1]
        return self.head(self.conv(x))


class Predictor(nn.Module):
    def __init__(self, dim: int = 128, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + act_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, dim),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class JEPA(nn.Module):
    def __init__(self, dim: int = 128, act_dim: int = 2, ema: float = 0.99, in_ch: int = 3):
        super().__init__()
        self.online = Encoder(dim, in_ch)
        self.predictor = Predictor(dim, act_dim)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False
        self.ema = ema
        self.dim = dim

    @torch.no_grad()
    def update_target(self):
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    def forward(self, img_t, a_t, img_t1):
        z_t = self.online(img_t)                      # online latent, current
        pred = self.predictor(z_t, a_t)               # predicted next latent
        with torch.no_grad():
            z_t1_target = self.target(img_t1)         # stop-grad target, next
        return z_t, pred, z_t1_target

    @torch.no_grad()
    def surprise(self, img_t, a_t, img_t1):
        """Per-sample latent prediction error = metacognitive surprise signal.

        Uses the same normalized metric the model was trained to minimize, so
        the signal is calibrated to the training objective."""
        z_t = self.online(img_t)
        pred = F.normalize(self.predictor(z_t, a_t), dim=-1)
        z_t1 = F.normalize(self.target(img_t1), dim=-1)
        return (pred - z_t1).pow(2).sum(dim=-1)        # (B,)


# ---- losses -------------------------------------------------------------
def prediction_loss(pred, target):
    # cosine-ish: normalize then MSE (scale-invariant, standard for JEPA targets)
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (pred - target).pow(2).sum(dim=-1).mean()


def vicreg_reg(z, var_coef: float = 25.0, cov_coef: float = 1.0, eps: float = 1e-4):
    """Variance hinge keeps each dim's std >= 1; covariance term decorrelates dims."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    cov_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
    return var_coef * var_loss + cov_coef * cov_loss, std.mean()


@torch.no_grad()
def effective_rank(z):
    """Participation ratio of covariance eigenvalues: (sum λ)^2 / sum λ^2."""
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    return (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item()
