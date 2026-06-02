"""Exp 2 (SPATIAL re-test, CORRECTED protocol): do Pi0.5's frozen VLM features
encode object POSITION?

History:
  v0 (pooled, train pert0 / eval OOD): 12.8cm in-dist, 24-31cm OOD.
  v1 (spatial keypoint, train pert0 / eval OOD): 2/5.5/8.6cm — BUT tied the
     predict-mean baseline on every split. label-spread in pert0 is only 1.8cm:
     the training data has ~no object-position variation, so the probe just
     learns the mean. The test was DEGENERATE (can't fit a regressor when the
     target is ~constant in training).

CORRECTED protocol (this file): POOL pert0+5+10 (together ~6-8cm of real position
variation), split by TRACE (held-out positions unseen), standardize targets, and
compare two read-outs of the SAME frozen VLM features against predict-mean:
  - SPATIAL : base-camera 14x14 patch grid -> soft-argmax keypoint head -> xyz.
  - POOLED  : mean-pooled grid -> MLP -> xyz  (the old representation, fair head).
predict-mean (always output train-mean position) is the bar both must beat.

Read: head-error << predict-mean => VLM features carry grasp-precise position ->
world encoder viable. head-error ~= predict-mean => features don't localize ->
the genuine perception wall (the VLM, not our handling).

Features are cached to <data>/exp2_spatial_cache.npz so head/split iterations
don't re-run the VLM.
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.contact_mpc.features.extractor import extract_spatial_features_from_dict
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import download
import rekey


GRID = 14
N_BASE = GRID * GRID       # 196 base-camera patch tokens
N_FRAMES = 6
BS = 8


def load_model(ckpt):
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10,
                               paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    params = _model.restore_params(download.maybe_download(ckpt), dtype=jnp.bfloat16)
    m = cfg.load(params); m.eval()
    return m


def resize224(imgs):
    imgs = imgs[:, ::-1, ::-1, :]
    out = jax.image.resize(imgs.astype(np.float32), (imgs.shape[0], 224, 224, 3), "bilinear")
    return np.asarray(jnp.clip(out, 0, 255).astype(jnp.uint8))


def extract_for_dir(model, files, tid_start):
    bases, wrists, labels, tids = [], [], [], []
    for ti_, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        imgs, wrist, it = d["image"], d["wrist_image"], d["image_t"]
        if imgs.shape[0] == 0:
            continue
        k = min(N_FRAMES, imgs.shape[0])
        names = list(d["object_names"]); ti = rekey.select_target_object(d["object_pos"], names)
        it = np.clip(it[:k], 0, d["object_pos"].shape[0] - 1)
        bases.append(resize224(imgs[:k])); wrists.append(resize224(wrist[:k]))
        labels.append(d["object_pos"][it, ti])
        tids.append(np.full(k, tid_start + ti_, np.int32))
    base = np.concatenate(bases); wr = np.concatenate(wrists)
    lab = np.concatenate(labels); tid = np.concatenate(tids)
    grids = []
    for b in range(0, base.shape[0], BS):
        bb, ww = base[b:b+BS], wr[b:b+BS]
        pad = BS - bb.shape[0]
        if pad:
            bb = np.concatenate([bb, np.repeat(bb[-1:], pad, 0)])
            ww = np.concatenate([ww, np.repeat(ww[-1:], pad, 0)])
        data = {
            "image": {"base_0_rgb": bb, "left_wrist_0_rgb": ww, "right_wrist_0_rgb": np.zeros_like(bb)},
            "image_mask": {k_: np.ones(BS, bool) for k_ in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            "state": np.zeros((BS, 32), np.float32),
        }
        prefix_out, _ = extract_spatial_features_from_dict(model, data)
        g = prefix_out[:, :N_BASE]
        grids.append(g[: bb.shape[0] - pad] if pad else g)
        if (b // BS) % 10 == 0:
            print(f"    batch {b//BS}/{(base.shape[0]+BS-1)//BS}", flush=True)
    return np.concatenate(grids), lab, tid


def get_features(args):
    cache = pathlib.Path(args.data) / "exp2_spatial_cache.npz"
    if cache.exists() and not args.refresh:
        print(f"loading cached features {cache}", flush=True)
        z = np.load(cache)
        return z["X"], z["Y"], z["tid"]
    model = load_model(args.ckpt)
    pat = lambda pert: sorted(glob.glob(str(pathlib.Path(args.data) / f"pert{pert}" /
                              f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    Xs, Ys, Ts = [], [], []
    base = 0
    for pert in (0, 5, 10):
        print(f"extracting pert{pert} ...", flush=True)
        X, Y, T = extract_for_dir(model, pat(pert), base)
        Xs.append(X); Ys.append(Y); Ts.append(T); base = int(T.max()) + 1
    X = np.concatenate(Xs); Y = np.concatenate(Ys).astype(np.float32); tid = np.concatenate(Ts)
    np.savez(cache, X=X, Y=Y, tid=tid)
    print(f"cached {X.shape} -> {cache}", flush=True)
    return X, Y, tid


import torch
torch.set_num_threads(8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class KeypointHead(torch.nn.Module):
    """Spatial soft-argmax: grid -> K heatmaps -> K (u,v) -> MLP -> xyz."""
    def __init__(self, D, K=16):
        super().__init__()
        self.detect = torch.nn.Linear(D, K)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * K, 128), torch.nn.SiLU(),
                                       torch.nn.Linear(128, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))
        ys, xs = torch.meshgrid(torch.linspace(0, 1, GRID), torch.linspace(0, 1, GRID), indexing="ij")
        self.register_buffer("xs", xs.reshape(-1)); self.register_buffer("ys", ys.reshape(-1))

    def forward(self, X):                                  # X [B,196,D]
        attn = torch.softmax(self.detect(X), dim=1)        # [B,196,K]
        u = (attn * self.xs[None, :, None]).sum(1)
        v = (attn * self.ys[None, :, None]).sum(1)
        return self.mlp(torch.cat([u, v], dim=1))


class PooledHead(torch.nn.Module):
    """Mean-pool the grid -> MLP -> xyz (the old representation, fair head)."""
    def __init__(self, D, **kw):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))

    def forward(self, X):
        return self.mlp(X.mean(1))


class AttnPoolHead(torch.nn.Module):
    """Learned-query cross-attention pool (no peak assumption) -> MLP -> xyz."""
    def __init__(self, D, n_query=8, dim=256):
        super().__init__()
        self.proj = torch.nn.Linear(D, dim)
        self.q = torch.nn.Parameter(torch.randn(n_query, dim) * 0.02)
        self.attn = torch.nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_query * dim, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 3))

    def forward(self, X):                                  # X [B,196,D]
        kv = self.proj(X)
        q = self.q[None].expand(X.shape[0], -1, -1)
        out, _ = self.attn(q, kv, kv)                      # [B,n_query,dim]
        return self.mlp(out.reshape(out.shape[0], -1))


def train_eval(HeadCls, name, Xtr, Ytr, Xte, Yte, epochs=1500, **kw):
    head = HeadCls(Xtr.shape[-1], **kw).to(DEV)
    Xtr_t = torch.tensor(Xtr, device=DEV); Xte_t = torch.tensor(Xte, device=DEV)
    Yt = torch.tensor(Ytr, device=DEV)
    mu, sd = Yt.mean(0), Yt.std(0) + 1e-6
    opt = torch.optim.Adam(head.parameters(), 1e-3)
    for ep in range(epochs):
        opt.zero_grad()
        loss = (((head(Xtr_t) - mu) / sd - (Yt - mu) / sd) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(Xte_t).cpu().numpy()
    err = np.linalg.norm(pred - Yte, axis=-1)
    ax = np.abs(pred - Yte).mean(0)
    return err.mean(), np.median(err), ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, default=3)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero/params")
    p.add_argument("--data", default="data/keystone")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, Y, tid = get_features(args)
    rng = np.random.default_rng(args.seed)
    utid = np.unique(tid); rng.shuffle(utid)
    n_te = max(1, int(len(utid) * 0.2))
    te_ids = set(utid[:n_te].tolist())
    te = np.array([t in te_ids for t in tid]); tr = ~te
    print(f"\npooled {X.shape}, {len(utid)} traces -> train {tr.sum()} / test {te.sum()} "
          f"({len(utid)-n_te}/{n_te} traces)", flush=True)

    Ytr, Yte = Y[tr], Y[te]
    spread = np.linalg.norm(Yte - Ytr.mean(0), axis=-1)            # predict-mean error on held-out
    print(f"label-spread (train): {np.linalg.norm(Ytr - Ytr.mean(0), axis=-1).mean()*100:.1f}cm  "
          f"per-axis std {Ytr.std(0).round(3)}", flush=True)
    print(f"predict-mean (held-out): mean {spread.mean()*100:.1f}cm  median {np.median(spread)*100:.1f}cm  "
          f"per-axis {np.abs(Yte-Ytr.mean(0)).mean(0).round(3)}\n", flush=True)

    print(f"device={DEV}\n", flush=True)
    runs = [
        ("keypoint K=16", KeypointHead, {"K": 16}),
        ("keypoint K=32", KeypointHead, {"K": 32}),
        ("attn-pool q=8", AttnPoolHead, {}),
        ("pooled MLP", PooledHead, {}),
    ]
    results = []
    for name, cls, kw in runs:
        print(f"training {name} ...", flush=True)
        m, md, ax = train_eval(cls, name, X[tr], Ytr, X[te], Yte, **kw)
        results.append((name, m, md, ax))

    print("\n=== EXP 2 CORRECTED (pooled data, trace-split): object-position error (cm) ===", flush=True)
    print(f"  {'predict-mean baseline':22s}: mean {spread.mean()*100:5.1f}  median {np.median(spread)*100:5.1f}", flush=True)
    for name, m, md, ax in results:
        print(f"  {name:22s}: mean {m*100:5.1f}  median {md*100:5.1f}  per-axis(cm) {(ax*100).round(1)}", flush=True)
    best = min(m for _, m, _, _ in results)
    bestname = min(results, key=lambda r: r[1])[0]
    verdict = ("LOCALIZES (beats predict-mean) -> world encoder viable" if best < 0.6 * spread.mean()
               else "TIES predict-mean -> features do NOT localize -> perception wall")
    print(f"\nVERDICT: best={bestname} {best*100:.1f}cm vs predict-mean {spread.mean()*100:.1f}cm -> {verdict}", flush=True)
    print("SPATIAL_PROBE_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
