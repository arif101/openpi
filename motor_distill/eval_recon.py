"""Teacher-forced reconstruction — clean manifold-transfer gate.

A0 (closed-loop g-replay) conflates two failure modes: (a) the manifold didn't
transfer, vs (b) open-loop replay feeds stale targets to a diverging head. This
isolates (a): feed the head Pi0.5's REAL (g, proprio) at each recorded step and
ask whether it predicts Pi0.5's actual chunk. No sim, no divergence.

Metrics (per held-out trace, pooled):
  dir_cos : cosine sim between predicted and true FIRST-action EE position-delta
            (does the head move the EE the right way? 1=perfect, 0=random)
  full_mse: MSE over the whole chunk (raw action units)
  grip_acc: fraction of first-action gripper signs that match
If dir_cos is high but A0 closed-loop is low -> the manifold transferred and the
gap is the open-loop confound (build target_gen). If dir_cos is low -> the
manifold/distillation is genuinely weak.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import head as H
import rekey

torch.set_num_threads(min(8, os.cpu_count() or 8))


def load_head(ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    net = H.MotorHead(horizon=ck["horizon"], hidden=ck.get("hidden", 256))
    net.load_state_dict(ck["state_dict"]); net.equivariant = ck["equivariant"]; net.eval()
    return net, ck["horizon"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--samples", type=int, default=4, help="avg flow samples to reduce noise")
    args = p.parse_args()
    net, _ = load_head(args.ckpt)

    pat = f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz"
    files = sorted(glob.glob(str(pathlib.Path(args.trace_dir) / pat)))[: args.n]
    cos_all, mse_all, grip_all = [], [], []
    for f in files:
        o = rekey.build_pairs(f, rekey.RekeyConfig(horizon=args.horizon))
        gp, gq = torch.tensor(o["g_pos"]), torch.tensor(o["g_quat"])
        pr, ch = torch.tensor(o["proprio"]), torch.tensor(o["chunk"])
        op, oq = torch.tensor(o["obj_pos"]), torch.tensor(o["obj_quat"])
        with torch.no_grad():
            preds = torch.stack([net.sample(gp, gq, pr, op, oq, steps=10)
                                 for _ in range(args.samples)]).mean(0)   # [N,H,7] mean
        # first-action EE position-delta direction
        pd, td = preds[:, 0, :3], ch[:, 0, :3]
        cos = (pd * td).sum(-1) / (pd.norm(dim=-1) * td.norm(dim=-1) + 1e-8)
        cos_all.append(cos.numpy())
        mse_all.append(((preds - ch) ** 2).mean().item())
        grip_all.append((torch.sign(preds[:, 0, 6]) == torch.sign(ch[:, 0, 6])).float().mean().item())
    cos = np.concatenate(cos_all)
    print(f"task {args.task_idx} [{'equiv' if net.equivariant else 'plain'}] "
          f"n={len(files)} traces, {len(cos)} steps:")
    print(f"  dir_cos  mean={cos.mean():.3f}  median={np.median(cos):.3f}  "
          f"frac>0.5={np.mean(cos > 0.5):.2f}")
    print(f"  full_mse {np.mean(mse_all):.4f}   grip_acc {np.mean(grip_all):.3f}")


if __name__ == "__main__":
    main()
