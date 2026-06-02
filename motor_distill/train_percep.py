"""Exp 3 prerequisite: train + save the perception head P (pooled MLP on Pi0.5
frozen spatial features -> object xyz). This is the world encoder we plug into
the NDP closed loop. Trained on the cached features (all perturbation levels, so
it learns to TRACK position, not just predict the in-dist mean).

Saves data/keystone/heads/percep_t3.pt (state_dict + arch).
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

    def forward(self, X):                       # X [B,196,D]
        return self.mlp(X.mean(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/keystone")
    p.add_argument("--epochs", type=int, default=3000)
    args = p.parse_args()
    z = np.load(pathlib.Path(args.data) / "exp2_spatial_cache.npz")
    X, Y = z["X"], z["Y"].astype(np.float32)
    head = PooledHead(X.shape[-1]).to(DEV)
    Xt = torch.tensor(X, device=DEV); Yt = torch.tensor(Y, device=DEV)
    mu, sd = Yt.mean(0), Yt.std(0) + 1e-6
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = (((head(Xt) - mu) / sd - (Yt - mu) / sd) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        err = np.linalg.norm(head(Xt).cpu().numpy() - Y, axis=-1)
    print(f"trained P on {X.shape[0]} frames; fit err mean {err.mean()*100:.1f}cm", flush=True)
    out = pathlib.Path(args.data) / "heads"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "D": X.shape[-1], "arch": "pooled"},
               out / "percep_t3.pt")
    print(f"saved {out}/percep_t3.pt", flush=True)
    print("PERCEP_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
