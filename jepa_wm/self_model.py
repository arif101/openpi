"""Self-model channel: predict the agent's OWN realized motion from its command.

  g(proprio_t, action_t) -> realized pusher delta (dpx, dpy)

The pusher is directly actuated, so this is near-deterministic on normal data —
a high-SNR self-model. The self-residual ‖predicted_delta - realized_delta‖ is the
toy analog of our real commanded-vs-realized end-effector ε (AUROC 0.97 on
execution-stuck). Trains in seconds from the existing npz, no anomalies seen.

Note vs the real-robot reachability field (which FAILED on deployment because it
excluded scene objects): here the pusher is a heavy position-controlled slider whose
motion is object-independent, so the self-model is clean. The failure mode there was
(embodiment, scene) coupling; a true self-model must be conditioned only on what it
can actually predict.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2)
        )

    def forward(self, feat):
        return self.net(feat)


def build_self_data(path):
    d = np.load(path)
    prop = d["proprio"]  # (N,T+1,4) = px,py,vx,vy
    acts = d["actions"]  # (N,T,2)
    N, T1 = prop.shape[0], prop.shape[1]
    T = T1 - 1
    feat, tgt = [], []
    for r in range(N):
        for t in range(T):
            feat.append(np.concatenate([prop[r, t], acts[r, t]]))
            tgt.append(prop[r, t + 1, :2] - prop[r, t, :2])  # realized pusher delta
    return np.array(feat, np.float32), np.array(tgt, np.float32)


def train_self_model(train_npz, out, epochs=200, device="cpu"):
    X, Y = build_self_data(train_npz)
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    model = SelfModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i : i + 1024]
            pred = model(Xt[idx])
            loss = (pred - Yt[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        res = (model(Xt) - Yt).pow(2).sum(-1).sqrt()
    print(f"self-model fit: mean residual {res.mean().item():.5f} m  (target std {Y.std():.4f})")
    torch.save({"model": model.state_dict()}, out)
    return model


def self_residual(model, proprio, action, realized_delta, device="cpu"):
    """Per-sample ‖predicted_delta - realized_delta‖ (meters)."""
    feat = torch.tensor(np.concatenate([proprio, action], -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(feat).cpu().numpy()
    return np.linalg.norm(pred - realized_delta, axis=-1)


if __name__ == "__main__":
    train_self_model("jepa_wm/data/pusher_train.npz", "jepa_wm/data/self_model.pt")
