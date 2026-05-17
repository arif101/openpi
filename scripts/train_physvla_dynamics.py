"""Train the PhysVLA latent dynamics model + physics aux heads.

Single GPU, PyTorch. Reads per-step physics-trace NPZs produced by
run_reason_v3_mppi.py --log-physics-traces. Trains:

  encoder(image, proprio) → z
  dynamics(z_t, action_t) → z_{t+1}_pred
  aux heads supervised on contact wrench, object poses, slip velocity

Acceptance gate (training only — value-add eval comes later):
  After training, the latent should *linearly probe* physics quantities at
  high R² (>0.7 on held-out trajectories). If yes, the latent has encoded
  physics; if no, the model is undercapacity or the supervision is too weak.

Usage:
    PYTHONPATH=src uv run python3 scripts/train_physvla_dynamics.py \\
        --traces-dir data/contact_mpc/physvla_traces \\
        --output-dir data/physvla/dynamics_v1 \\
        --num-epochs 30 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from openpi.physvla.dataset import PhysVLAFrameDataset
from openpi.physvla.model import PhysVLAConfig, PhysVLAModel, physvla_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-objects", type=int, default=12)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--w-dyn", type=float, default=1.0)
    p.add_argument("--w-wrench", type=float, default=0.5)
    p.add_argument("--w-pose", type=float, default=1.0)
    p.add_argument("--w-slip", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def split_dataset(ds: PhysVLAFrameDataset, val_frac: float, seed: int = 0):
    """Episode-aware split. Hold out whole episodes for val, not random frames."""
    rng = np.random.default_rng(seed)
    n_eps = len(ds.npz_paths)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(n_eps * val_frac))
    val_files = set(int(x) for x in perm[:n_val])

    train_idx = [i for i, (fi, _) in enumerate(ds.index) if fi not in val_files]
    val_idx = [i for i, (fi, _) in enumerate(ds.index) if fi in val_files]
    return train_idx, val_idx


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PhysVLA traces from {args.traces_dir} ...")
    ds = PhysVLAFrameDataset(args.traces_dir, n_objects=args.n_objects)

    train_idx, val_idx = split_dataset(ds, args.val_frac)
    print(f"Split: train {len(train_idx)}  val {len(val_idx)}")
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    # Probe one sample to compute proprio dim.
    sample = ds[0]
    proprio_in_dim = int(sample["proprio_t"].shape[0])
    print(f"Proprio in-dim: {proprio_in_dim}")

    cfg = PhysVLAConfig(latent_dim=args.latent_dim, n_objects=args.n_objects)
    model = PhysVLAModel(cfg, proprio_in_dim=proprio_in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    metrics_log: list[dict] = []
    best_val = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        ep_metrics: list[dict] = []
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, parts = physvla_loss(
                model, batch,
                w_dyn=args.w_dyn, w_wrench=args.w_wrench,
                w_pose=args.w_pose, w_slip=args.w_slip,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_metrics.append(parts)
            if it % args.log_every == 0:
                print(f"  epoch {epoch:>2d} it {it:>4d}  "
                      f"L_tot={parts['L_total']:.4f} "
                      f"L_dyn={parts['L_dyn']:.4f} "
                      f"L_w={parts['L_wrench']:.4f} "
                      f"L_p={parts['L_pose']:.4f} "
                      f"L_s={parts['L_slip']:.4f}", flush=True)
        sched.step()

        # Validation pass
        model.eval()
        val_parts: list[dict] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                _, parts = physvla_loss(
                    model, batch,
                    w_dyn=args.w_dyn, w_wrench=args.w_wrench,
                    w_pose=args.w_pose, w_slip=args.w_slip,
                )
                val_parts.append(parts)
        agg = {f"train_{k}": float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0]}
        agg.update({f"val_{k}":   float(np.mean([m[k] for m in val_parts])) for k in val_parts[0]})
        agg["epoch"] = epoch
        agg["wall_s"] = time.time() - t0
        metrics_log.append(agg)
        print(f"[epoch {epoch}] train_L={agg['train_L_total']:.4f}  "
              f"val_L={agg['val_L_total']:.4f}  ({agg['wall_s']:.1f}s)", flush=True)

        # Save best by val loss.
        if agg["val_L_total"] < best_val:
            best_val = agg["val_L_total"]
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "proprio_in_dim": proprio_in_dim,
                "epoch": epoch,
                "val_L_total": best_val,
            }, out_dir / "physvla_dynamics_best.pt")
        (out_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))

    # Final save.
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "proprio_in_dim": proprio_in_dim,
        "epoch": args.num_epochs - 1,
        "val_L_total": agg["val_L_total"],
    }, out_dir / "physvla_dynamics_final.pt")
    print(f"\nDone. Best val loss: {best_val:.4f}. Saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
