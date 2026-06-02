"""Exp 2 decision-maker: learning curve on the CACHED VLM features.

The corrected probe showed features DO encode position (train fit 1.3cm) but
generalize to only 3.7cm on held-out traces — a data-starvation gap (77 unique
positions, 4cm spread). This settles whether MORE position-varied data would
close the gap: train the best head (pooled MLP) on an increasing number of
training traces and report held-out error. Steeply decreasing at 77 traces ->
collect more data. Flat -> data won't help, it's the features.

Held-out test traces are FIXED across all training sizes (apples-to-apples).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

torch.set_num_threads(8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class PooledHead(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))

    def forward(self, X):
        return self.mlp(X.mean(1))


def fit_eval(Xtr, Ytr, Xte, Yte, epochs=1500):
    head = PooledHead(Xtr.shape[-1]).to(DEV)
    Xtr_t = torch.tensor(Xtr, device=DEV); Xte_t = torch.tensor(Xte, device=DEV)
    Yt = torch.tensor(Ytr, device=DEV)
    mu, sd = Yt.mean(0), Yt.std(0) + 1e-6
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (((head(Xtr_t) - mu) / sd - (Yt - mu) / sd) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(Xte_t).cpu().numpy(); ptr = head(Xtr_t).cpu().numpy()
    return (np.linalg.norm(pred - Yte, axis=-1).mean(), np.median(np.linalg.norm(pred - Yte, axis=-1)),
            np.linalg.norm(ptr - Ytr, axis=-1).mean())


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
    te_ids = set(utid[:n_te].tolist()); train_ids = utid[n_te:]
    te = np.array([t in te_ids for t in tid])
    Xte, Yte = X[te], Y[te]
    base = np.linalg.norm(Yte - Y[~te].mean(0), axis=-1)
    print(f"{len(utid)} traces, held-out {n_te} ({te.sum()} frames). "
          f"predict-mean test {base.mean()*100:.1f}cm / med {np.median(base)*100:.1f}cm\n", flush=True)
    print(f"{'n_train_traces':>14s}  {'TRAIN':>6s}  {'test-mean':>9s}  {'test-med':>8s}", flush=True)
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(4, int(len(train_ids) * frac))
        use = set(train_ids[:k].tolist())
        tr = np.array([t in use for t in tid])
        tm, tmd, trn = fit_eval(X[tr], Y[tr], Xte, Yte)
        print(f"{k:>14d}  {trn*100:5.1f}  {tm*100:8.1f}  {tmd*100:7.1f}", flush=True)
    print("\nRead: test-error still dropping at max traces -> MORE DATA closes the gap (collect). "
          "Flat -> data won't help (features-limited).", flush=True)
    print("CURVE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
