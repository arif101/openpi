"""Stratified contact-conditioned probe of a trained PhysVLA latent.

Why stratified: the val set is dominated by no-contact, no-motion frames.
A model that predicts "wrench = 0, pose = current pose, slip = 0" achieves
R² > 0.9 aggregate while being useless on contact moments. The whole
PhysVLA thesis depends on getting contact and slip RIGHT, not on average
performance over a mostly-stationary majority.

The gate that decides whether to invest in Week 3 (action refinement
built on this latent) is contact-conditioned R², not aggregate R².

Outputs a table with both views, plus a degenerate-baseline reference.
PhysVLA passes only if it's meaningfully above the predict-zero baseline
on the contact-conditioned slices.

Usage:
    PYTHONPATH=src uv run python3 scripts/probe_physvla_latent.py \\
        --checkpoint data/physvla/dynamics_v2/physvla_dynamics_best.pt \\
        --traces-dir data/contact_mpc/physvla_traces
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

from openpi.physvla.dataset import PhysVLAFrameDataset
from openpi.physvla.model import PhysVLAConfig, PhysVLAModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--traces-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Stratification thresholds. "Contact" = wrench magnitude above the τ_w
    # quantile of the dataset's wrench-magnitude distribution. Using a
    # quantile (top decile) is more robust than a fixed N threshold across
    # tasks/scenes.
    p.add_argument("--contact-quantile", type=float, default=0.9,
                   help="Wrench magnitude quantile that defines 'contact'. "
                        "Default 0.9 = top 10%% of frames by wrench magnitude.")
    p.add_argument("--motion-threshold", type=float, default=0.005,
                   help="Per-frame per-object displacement (m) above which "
                        "the frame counts as 'in motion'. Default 5mm.")
    p.add_argument("--slip-quantile", type=float, default=0.9,
                   help="Slip magnitude quantile that defines 'slipping'.")
    return p.parse_args()


def r2(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Vector R² (treat each output dim as a regression target)."""
    if mask is not None:
        if mask.sum() < 2:
            return float("nan")
        preds = preds[mask]
        targets = targets[mask]
    ss_res = float(np.sum((preds - targets) ** 2))
    ss_tot = float(np.sum((targets - targets.mean(axis=0)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def predict_zero(preds_shape: tuple) -> np.ndarray:
    """Degenerate baseline: predict zeros."""
    return np.zeros(preds_shape, dtype=np.float32)


def predict_mean(targets: np.ndarray) -> np.ndarray:
    """Degenerate baseline: predict dataset mean."""
    mean = targets.mean(axis=0, keepdims=True)
    return np.broadcast_to(mean, targets.shape).copy()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = PhysVLAConfig(**ckpt["cfg"])
    model = PhysVLAModel(cfg, proprio_in_dim=ckpt["proprio_in_dim"]).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    print(f"  loaded epoch={ckpt.get('epoch')} val_L_total={ckpt.get('val_L_total')}")

    print(f"\nLoading val dataset from {args.traces_dir}...")
    ds = PhysVLAFrameDataset(args.traces_dir, n_objects=cfg.n_objects)
    n_eps = len(ds.npz_paths)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_eps)
    n_val = max(1, int(n_eps * args.val_frac))
    val_files = set(int(x) for x in perm[:n_val])
    val_idx = [i for i, (fi, _) in enumerate(ds.index) if fi in val_files]
    val_ds = torch.utils.data.Subset(ds, val_idx)
    print(f"  val pairs: {len(val_ds)}")
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                          shuffle=False, num_workers=0)

    # Collect predictions + targets across val set.
    wrench_pred, wrench_target = [], []
    pose_pred, pose_target = [], []
    slip_pred, slip_target = [], []
    object_displacements = []   # per-frame max object displacement, t→t+1

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_t = model.encode(batch["image_t"], batch["proprio_t"])
            z_tp1_pred = model.step(z_t, batch["action_t"])
            preds = model.aux_predictions(z_tp1_pred)

            wrench_pred.append(preds["wrench"].cpu().numpy())
            wrench_target.append(batch["wrench_tp1"].cpu().numpy())

            B = batch["pose_tp1"].shape[0]
            pose_target_flat = batch["pose_tp1"].reshape(B, -1).cpu().numpy()
            n_dims = pose_target_flat.shape[-1]
            pose_pred.append(preds["pose"][:, :n_dims].cpu().numpy())
            pose_target.append(pose_target_flat)

            slip_pred.append(preds["slip"].cpu().numpy())
            slip_target.append(batch["slip_tp1"].cpu().numpy())

            # Per-frame object displacement = max ||obj_pos(t+1) - obj_pos(t)||
            # over objects. Object positions are in the proprio vector at the
            # n_obj × 3 block. Reconstruct from proprio_t / proprio_tp1.
            proprio_t = batch["proprio_t"].cpu().numpy()
            proprio_tp1 = batch["proprio_tp1"].cpu().numpy()
            # robot_qpos(9) + robot_qvel(9) + ee_pos(3) + ee_quat(4) + gripper(2) = 27
            # then object_pos block = n_obj * 3
            n_obj = cfg.n_objects
            obj_pos_t = proprio_t[:, 27:27 + n_obj * 3].reshape(B, n_obj, 3)
            obj_pos_tp1 = proprio_tp1[:, 27:27 + n_obj * 3].reshape(B, n_obj, 3)
            disp = np.linalg.norm(obj_pos_tp1 - obj_pos_t, axis=-1)  # [B, n_obj]
            max_disp = disp.max(axis=-1)                              # [B]
            object_displacements.append(max_disp)

    wrench_pred = np.concatenate(wrench_pred)
    wrench_target = np.concatenate(wrench_target)
    pose_pred = np.concatenate(pose_pred)
    pose_target = np.concatenate(pose_target)
    slip_pred = np.concatenate(slip_pred)
    slip_target = np.concatenate(slip_target)
    object_displacements = np.concatenate(object_displacements)

    print(f"\n=== Aggregate R² (ALL val frames) — predict-zero baseline ===")
    print(f"  wrench  predict-zero: {r2(predict_zero(wrench_target.shape), wrench_target):.4f}")
    print(f"  pose    predict-mean: {r2(predict_mean(pose_target), pose_target):.4f}")
    print(f"  slip    predict-zero: {r2(predict_zero(slip_target.shape), slip_target):.4f}")

    print(f"\n=== Aggregate R² (ALL val frames) — model ===")
    print(f"  wrench:  {r2(wrench_pred, wrench_target):.4f}")
    print(f"  pose:    {r2(pose_pred, pose_target):.4f}")
    print(f"  slip:    {r2(slip_pred, slip_target):.4f}")

    # Stratified slices.
    wrench_mag = np.linalg.norm(wrench_target, axis=-1)
    slip_mag = np.linalg.norm(slip_target, axis=-1)

    tau_w = float(np.quantile(wrench_mag, args.contact_quantile))
    tau_s = float(np.quantile(slip_mag, args.slip_quantile))
    contact_mask = wrench_mag > tau_w
    motion_mask = object_displacements > args.motion_threshold
    slip_mask = slip_mag > tau_s

    print(f"\n=== Stratification ===")
    print(f"  contact: ‖wrench‖ > {tau_w:.4f}  → {contact_mask.sum():>6d} / {len(contact_mask)} frames ({contact_mask.mean()*100:.1f}%)")
    print(f"  motion:  obj displacement > {args.motion_threshold*1000:.0f}mm → {motion_mask.sum():>6d} ({motion_mask.mean()*100:.1f}%)")
    print(f"  slip:    ‖slip‖ > {tau_s:.4f} → {slip_mask.sum():>6d} ({slip_mask.mean()*100:.1f}%)")

    print(f"\n=== Contact-conditioned R² — predict-zero baseline ===")
    print(f"  wrench (contact frames only): {r2(predict_zero(wrench_target.shape), wrench_target, contact_mask):.4f}")
    print(f"  pose   (motion frames only):  {r2(predict_mean(pose_target), pose_target, motion_mask):.4f}")
    print(f"  slip   (slip frames only):    {r2(predict_zero(slip_target.shape), slip_target, slip_mask):.4f}")

    print(f"\n=== Contact-conditioned R² — model ===")
    r2_w_c = r2(wrench_pred, wrench_target, contact_mask)
    r2_p_m = r2(pose_pred, pose_target, motion_mask)
    r2_s_s = r2(slip_pred, slip_target, slip_mask)
    print(f"  wrench (contact frames only): {r2_w_c:.4f}    {'PASS' if r2_w_c > 0.6 else 'FAIL'}")
    print(f"  pose   (motion frames only):  {r2_p_m:.4f}    {'PASS' if r2_p_m > 0.6 else 'FAIL'}")
    print(f"  slip   (slip frames only):    {r2_s_s:.4f}    {'PASS' if r2_s_s > 0.5 else 'FAIL'}")

    # Multi-step rollout error on contact frames: roll the dynamics K steps
    # from a contact frame's z_t and compare aux predictions across the window.
    # We approximate by doing single-step rollouts then comparing the wrench
    # MAE specifically on contact frames vs non-contact.
    wrench_mae_contact = float(np.mean(np.abs(wrench_pred[contact_mask] - wrench_target[contact_mask])))
    wrench_mae_nocontact = float(np.mean(np.abs(wrench_pred[~contact_mask] - wrench_target[~contact_mask])))
    print(f"\n=== Wrench MAE — contact vs no-contact ===")
    print(f"  contact frames:    {wrench_mae_contact:.4f}")
    print(f"  no-contact frames: {wrench_mae_nocontact:.4f}")
    print(f"  ratio (contact/no): {wrench_mae_contact / max(wrench_mae_nocontact, 1e-9):.2f}x "
          f"({'good (≤2x)' if wrench_mae_contact <= 2 * wrench_mae_nocontact else 'concerning (>2x; model degrades on contact)'})")

    print(f"\n=== Verdict ===")
    passes = [r2_w_c > 0.6, r2_p_m > 0.6, r2_s_s > 0.5]
    if all(passes):
        print(f"  ✓ PROCEED to Week 3 — latent encodes physics on contact-conditioned slices.")
    elif sum(passes) >= 2:
        print(f"  ≈ MARGINAL — passes 2/3 contact-conditioned gates. Investigate weakest signal "
              f"before committing to Week 3.")
    else:
        print(f"  ✗ FAIL — latent does not encode contact-conditioned physics. Don't build "
              f"Week 3 refinement on it. Diagnose by (a) checking aux loss curves, "
              f"(b) scaling latent dim, (c) over-sampling contact frames in dataloader.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
