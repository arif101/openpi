"""Distill Pi0.5's motor manifold into the goal-relative head (keystone probe).

Offline flow-matching behaviour cloning on re-keyed SUCCESSFUL Pi0.5 rollouts:
load PHYS_OK baseline traces -> rekey to (g, proprio, chunk, obj_pose) pairs ->
fit a normalizer -> train MotorHead with an episode-held-out val split -> save.
Trains BOTH variants (plain + equivariant) so the eval harness can run the
Test-B ablation.

Usage:
  python train_distill.py --trace-dir data/keystone/pert0 --out ckpt/ \
      --horizon 16 --steps 8000
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

# Cap CPU threads: on many-core boxes torch oversubscribes (128 threads for a
# tiny MLP's matmuls) and thrashes on thread-sync, running ~10x slower. 8 is
# plenty for a 0.5M-param head.
torch.set_num_threads(min(8, os.cpu_count() or 8))

import head as H
import rekey


def load_dataset(trace_dir, horizon, pattern="PHYS_OK_*_baseline_*.npz"):
    """Pool re-keyed pairs from all successful baseline traces in a dir.
    Returns dict of torch tensors + per-pair episode id (for honest val split)."""
    files = sorted(glob.glob(str(pathlib.Path(trace_dir) / pattern)))
    if not files:
        # fall back: any PHYS_OK in the dir
        files = sorted(glob.glob(str(pathlib.Path(trace_dir) / "PHYS_OK_*.npz")))
    if not files:
        raise FileNotFoundError(f"no PHYS_OK traces in {trace_dir}")
    cols = {k: [] for k in ("g_pos", "g_quat", "proprio", "chunk", "obj_pos", "obj_quat")}
    ep_id = []
    kept, skipped = 0, 0
    for ei, f in enumerate(files):
        try:
            out = rekey.build_pairs(f, rekey.RekeyConfig(horizon=horizon))
        except Exception as e:
            skipped += 1
            continue
        if not out["success"]:
            skipped += 1
            continue
        n = out["g_pos"].shape[0]
        for k in cols:
            cols[k].append(out[k])
        ep_id.append(np.full(n, ei, dtype=np.int64))
        kept += 1
    data = {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in cols.items()}
    data["ep_id"] = torch.tensor(np.concatenate(ep_id))
    print(f"loaded {kept} episodes ({skipped} skipped), {data['g_pos'].shape[0]} pairs, "
          f"horizon={horizon}")
    return data


def episode_split(ep_id, val_frac=0.15, seed=0):
    eps = torch.unique(ep_id)
    g = torch.Generator().manual_seed(seed)
    perm = eps[torch.randperm(len(eps), generator=g)]
    n_val = max(1, int(len(eps) * val_frac)) if len(eps) >= 5 else 0
    val_eps = set(perm[:n_val].tolist())
    val_mask = torch.tensor([e.item() in val_eps for e in ep_id])
    if n_val == 0:                                   # too few eps (smoke) -> val=train
        return torch.ones_like(val_mask, dtype=torch.bool), torch.ones_like(val_mask, dtype=torch.bool)
    return ~val_mask, val_mask


def fit_normalizer(net, d, idx, equivariant, device):
    """Set net's cond/chunk normalizer from the TRAIN split (canonical space)."""
    net.equivariant = equivariant
    with torch.no_grad():
        cond, psi = H.cond_vector(d["g_pos"][idx], d["g_quat"][idx], d["proprio"][idx],
                                  d["obj_pos"][idx], d["obj_quat"][idx], equivariant)
        chunk_c = H.chunk_to_canonical(d["chunk"][idx], psi).reshape(idx.sum(), -1)
    net.cond_mean.copy_(cond.mean(0)); net.cond_std.copy_(cond.std(0).clamp_min(1e-4))
    net.chunk_mean.copy_(chunk_c.mean(0)); net.chunk_std.copy_(chunk_c.std(0).clamp_min(1e-4))


def train_one(d, equivariant, args, device):
    net = H.MotorHead(horizon=args.horizon, hidden=args.hidden).to(device)
    tr_mask, va_mask = episode_split(d["ep_id"], args.val_frac, args.seed)
    fit_normalizer(net, d, tr_mask, equivariant, device)
    net.equivariant = equivariant
    keys = ("g_pos", "g_quat", "proprio", "chunk", "obj_pos", "obj_quat")
    tr = {k: d[k][tr_mask].to(device) for k in keys}
    va = {k: d[k][va_mask].to(device) for k in keys}
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ntr = tr["g_pos"].shape[0]
    best = (1e9, None)
    tag = "equiv" if equivariant else "plain"
    for step in range(1, args.steps + 1):
        net.train()
        bi = torch.randint(0, ntr, (args.batch,), device=device)
        loss = net.loss(*(tr[k][bi] for k in keys))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.val_every == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                vl = torch.stack([net.loss(*(va[k] for k in keys)) for _ in range(4)]).mean().item()
            print(f"  [{tag}] step {step:5d}  train {loss.item():.4f}  val {vl:.4f}", flush=True)
            if vl < best[0]:
                best = (vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"head_{tag}.pt"
    torch.save({"state_dict": net.state_dict(), "horizon": args.horizon,
                "hidden": args.hidden, "equivariant": equivariant,
                "best_val": best[0]}, ckpt)
    print(f"  [{tag}] saved {ckpt}  best_val={best[0]:.4f}")
    return best[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--out", default="ckpt")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", nargs="+", default=["plain", "equiv"],
                   choices=["plain", "equiv"])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device={device}")
    d = load_dataset(args.trace_dir, args.horizon)
    for v in args.variants:
        train_one(d, v == "equiv", args, device)


if __name__ == "__main__":
    main()
