"""Option A: image-keypoint + low-DOF planar-geometry lift.

Hypothesis: the frozen-feature metric-xyz regression plateaus (3.7cm) because a
free MLP must LEARN the camera projection from 77 positions and can't. Fix: force
perception through a 2D image-location bottleneck (soft-argmax keypoints over the
14x14 grid) and lift to world with a GLOBAL LOW-DOF geometric map (linear /
homography, <=27 params) instead of a free MLP. A homography IS the exact
image->table-plane map, so it generalizes from few points and can't overfit.

We test it WITHOUT a pixel-perfect camera matrix: the keypoints are learned
through the world-(x,y[,z]) loss, and the low-DOF lift is fit jointly. This is
equivalent to the analytic-camera version for the generalization question, and
footgun-free. (Analytic camera matrix is deferred to deployment.)

Baselines in the same trace-split protocol: pooled MLP (3.7cm), deep-MLP keypoint
(4.2cm), predict-mean (5.5cm). Cache: data/keystone/exp2_spatial_cache.npz.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

torch.set_num_threads(8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 14


class KPLinear(torch.nn.Module):
    """soft-argmax K keypoints -> LOW-DOF linear lift -> world xyz."""
    def __init__(self, D, K=1):
        super().__init__()
        self.detect = torch.nn.Linear(D, K)
        self.lift = torch.nn.Linear(2 * K, 3)            # <=27 params; the geometry
        ys, xs = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1)); self.register_buffer("ys", ys.reshape(-1))

    def forward(self, X):
        attn = torch.softmax(self.detect(X), dim=1)
        u = (attn * self.xs[None, :, None]).sum(1)
        v = (attn * self.ys[None, :, None]).sum(1)
        return self.lift(torch.cat([u, v], dim=1))


class KPHomography(torch.nn.Module):
    """single soft-argmax keypoint -> 3x3 homography -> table (x,y); z = learned const."""
    def __init__(self, D, K=1):
        super().__init__()
        self.detect = torch.nn.Linear(D, 1)
        self.H = torch.nn.Parameter(torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]]))
        self.z = torch.nn.Parameter(torch.zeros(1))
        ys, xs = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1)); self.register_buffer("ys", ys.reshape(-1))

    def forward(self, X):
        attn = torch.softmax(self.detect(X), dim=1)
        u = (attn * self.xs[None, :, None]).sum(1)
        v = (attn * self.ys[None, :, None]).sum(1)
        uv1 = torch.cat([u, v, torch.ones_like(u)], dim=1)        # [B,3]
        p = uv1 @ self.H.T                                        # [B,3]
        xy = p[:, :2] / (p[:, 2:3] + 1e-6)
        z = self.z.expand(X.shape[0], 1)
        return torch.cat([xy, z], dim=1)


class PooledHead(torch.nn.Module):
    def __init__(self, D, **kw):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))

    def forward(self, X):
        return self.mlp(X.mean(1))


def train_eval(HeadCls, Xtr, Ytr, Xte, Yte, epochs=2500, lr=2e-3, **kw):
    head = HeadCls(Xtr.shape[-1], **kw).to(DEV)
    Xtr_t = torch.tensor(Xtr, device=DEV); Xte_t = torch.tensor(Xte, device=DEV)
    Yt = torch.tensor(Ytr, device=DEV)
    mu, sd = Yt.mean(0), Yt.std(0) + 1e-6
    opt = torch.optim.Adam(head.parameters(), lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (((head(Xtr_t) - mu) / sd - (Yt - mu) / sd) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(Xte_t).cpu().numpy(); ptr = head(Xtr_t).cpu().numpy()
    err = np.linalg.norm(pred - Yte, axis=-1)
    ax = np.abs(pred - Yte).mean(0)
    return np.linalg.norm(ptr - Ytr, axis=-1).mean(), err.mean(), np.median(err), ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/keystone")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    z = np.load(pathlib.Path(args.data) / "exp2_spatial_cache.npz")
    X, Y, tid = z["X"], z["Y"].astype(np.float32), z["tid"]
    rng = np.random.default_rng(args.seed)
    utid = np.unique(tid); rng.shuffle(utid)
    n_te = max(1, int(len(utid) * 0.2))
    te_ids = set(utid[:n_te].tolist())
    te = np.array([t in te_ids for t in tid]); tr = ~te
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    spread = np.linalg.norm(Yte - Ytr.mean(0), axis=-1)
    print(f"device={DEV}  train {tr.sum()} / test {te.sum()}  predict-mean {spread.mean()*100:.1f}cm / "
          f"med {np.median(spread)*100:.1f}cm\n", flush=True)

    runs = [
        ("KP->linear K=1", KPLinear, {"K": 1}),
        ("KP->linear K=2", KPLinear, {"K": 2}),
        ("KP->linear K=4", KPLinear, {"K": 4}),
        ("KP->homography", KPHomography, {}),
        ("pooled MLP (ref)", PooledHead, {}),
    ]
    print(f"  {'readout':18s}  {'TRAIN':>6s}  {'test-mean':>9s}  {'test-med':>8s}  per-axis(cm)", flush=True)
    best = (None, 1e9)
    for name, cls, kw in runs:
        trn, m, md, ax = train_eval(cls, Xtr, Ytr, Xte, Yte, **kw)
        print(f"  {name:18s}  {trn*100:5.1f}  {m*100:8.1f}  {md*100:7.1f}  {(ax*100).round(1)}", flush=True)
        if m < best[1]:
            best = (name, m)
    print(f"\n  predict-mean baseline: {spread.mean()*100:.1f}cm | metric-MLP ref: 3.7cm", flush=True)
    verdict = ("OPTION A WINS -> image-bottleneck + low-DOF geometry generalizes better"
               if best[1] < 0.028 else "no clear win over metric-MLP -> geometry lift insufficient")
    print(f"  best={best[0]} {best[1]*100:.1f}cm -> {verdict}", flush=True)
    print("OPTIONA_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
