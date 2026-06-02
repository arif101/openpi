"""Train the tiny JEPA and run the M1 gate.

M1 gate (both must pass):
  (a) anti-collapse: mean embedding std stays high AND effective rank >> 1.
  (b) learned structure: a linear probe recovers ball position from frozen latents
      (R^2 high) — V-JEPA-2's "3D structure from pixels" claim in miniature.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from jepa_wm.data import build_stacked_transitions, to_float
from jepa_wm.model import JEPA, effective_rank, prediction_loss, vicreg_reg


def linear_probe_r2(model, img, target, device, bs=512):
    """Frozen-encoder linear probe: ridge-fit z -> ball_pos, report R^2."""
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(img), bs):
            z = model.online(to_float(img[i : i + bs], device))
            zs.append(z.cpu())
    Z = torch.cat(zs).numpy()
    Y = target.numpy()
    n = len(Z)
    ntr = int(0.8 * n)
    Xtr, Ytr, Xte, Yte = Z[:ntr], Y[:ntr], Z[ntr:], Y[ntr:]
    Xtr1 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    Xte1 = np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1)
    lam = 1e-2
    W = np.linalg.solve(Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1]), Xtr1.T @ Ytr)
    pred = Xte1 @ W
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="jepa_wm/data/pusher_train.npz")
    p.add_argument("--val", default="jepa_wm/data/pusher_val.npz")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vicreg", type=float, default=1.0, help="VICReg weight (0 = pure EMA)")
    p.add_argument("--stack", type=int, default=2, help="frames stacked on channels (velocity if >1)")
    p.add_argument("--out", default="jepa_wm/data/jepa.pt")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  vicreg_weight={args.vicreg}  stack={args.stack}")

    img_t, img_t1, a, ball_t = build_stacked_transitions(args.train, args.stack)
    vimg_t, vimg_t1, va, vball = build_stacked_transitions(args.val, args.stack)
    n = len(img_t)
    print(f"train transitions: {n}  val: {len(vimg_t)}")

    model = JEPA(dim=args.dim, in_ch=3 * args.stack).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {nparams/1e6:.2f}M")
    opt = torch.optim.AdamW(
        list(model.online.parameters()) + list(model.predictor.parameters()), lr=args.lr, weight_decay=1e-4
    )

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot_pred = tot_std = tot_rank = nb = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i : i + args.bs]
            xt = to_float(img_t[idx], device)
            xt1 = to_float(img_t1[idx], device)
            at = a[idx].to(device)
            z_t, pred, z_t1_tgt = model(xt, at, xt1)
            pl = prediction_loss(pred, z_t1_tgt)
            vl, std = vicreg_reg(z_t)
            loss = pl + args.vicreg * vl
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target()
            tot_pred += pl.item(); tot_std += std.item()
            tot_rank += effective_rank(z_t.detach()); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep+1:3d}  pred_loss {tot_pred/nb:.4f}  emb_std {tot_std/nb:.3f}  "
                  f"eff_rank {tot_rank/nb:.1f}/{args.dim}  ({(ep+1)/(time.time()-t0):.2f} ep/s)")

    # ---- M1 gate ----
    print("\n=== M1 GATE ===")
    model.eval()
    with torch.no_grad():
        z_sample = model.online(to_float(vimg_t[:1024], device))
    std = z_sample.std(0).mean().item()
    rank = effective_rank(z_sample)
    r2 = linear_probe_r2(model, vimg_t, vball, device)
    collapse_ok = std > 0.3 and rank > 10
    structure_ok = r2 > 0.8
    print(f"(a) anti-collapse : emb_std={std:.3f}  eff_rank={rank:.1f}/{args.dim}  -> {'PASS' if collapse_ok else 'FAIL'}")
    print(f"(b) structure     : ball-pos linear-probe R^2={r2:.3f}  -> {'PASS' if structure_ok else 'FAIL'}")
    print(f"M1 {'PASS' if (collapse_ok and structure_ok) else 'FAIL'}")

    torch.save({"model": model.state_dict(), "dim": args.dim, "args": vars(args)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
