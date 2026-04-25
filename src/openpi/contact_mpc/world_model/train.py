"""Training loop for the latent world model.

Trains on (h_t, action_chunk, h_{t+H}) triples with MSE loss.
Supports:
- Horizon sweep (H=3, 5, 10)
- Size sweep (small ~1M, medium ~5M, large ~20M)
- Contact-only training variant (for KS2)
- Train/val split with early stopping
"""

from __future__ import annotations

import logging
import pathlib
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openpi.contact_mpc.features.dataset import FeatureDataset
from openpi.contact_mpc.world_model.architecture import (
    LatentWorldModel,
    WorldModelConfig,
    SIZE_CONFIGS,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def build_dataset_for_horizon(
    features: FeatureDataset,
    horizon: int,
    contact_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (h_t, action_chunk[:H], h_{t+H}) from the feature dataset.

    The stored dataset was built with a specific H (e.g., 10). To train
    with a smaller H, we truncate the action chunks and look up the
    corresponding future hidden state at offset H instead of the stored offset.

    For H smaller than the stored horizon, we need the original feature
    dataset to have been built with the larger H, and we reconstruct
    shorter-horizon triples from it.

    Args:
        features: The extracted feature dataset.
        horizon: Target horizon H for this training run.
        contact_only: If True, filter to contact-phase timesteps only.

    Returns:
        Tuple of (hidden_states, action_chunks, future_hidden_states).
    """
    stored_H = features.horizon

    if horizon > stored_H:
        raise ValueError(
            f"Requested horizon {horizon} > stored horizon {stored_H}. "
            "Re-extract features with a larger horizon."
        )

    if horizon == stored_H:
        h = features.hidden_states
        a = features.action_chunks
        fh = features.future_hidden_states
        mask = features.is_contact if contact_only else np.ones(len(h), dtype=bool)
    else:
        # For shorter horizons, we need to reconstruct triples.
        # The stored h_{t+stored_H} is at offset stored_H, but we need h_{t+horizon}.
        # We can get this from the dataset: h_{t+horizon} = hidden_states of the
        # record that starts at timestep (t + horizon) in the same episode.
        #
        # Build a lookup: (episode_id, timestep) → index
        lookup = {}
        for i in range(len(features.hidden_states)):
            key = (int(features.episode_ids[i]), int(features.timesteps[i]))
            lookup[key] = i

        h_list, a_list, fh_list = [], [], []
        for i in range(len(features.hidden_states)):
            ep_id = int(features.episode_ids[i])
            t = int(features.timesteps[i])
            future_key = (ep_id, t + horizon)
            if future_key in lookup:
                if contact_only and not features.is_contact[i]:
                    continue
                h_list.append(features.hidden_states[i])
                a_list.append(features.action_chunks[i, :horizon, :])
                fh_list.append(features.hidden_states[lookup[future_key]])
                # Note: we use hidden_states (not future_hidden_states) of the
                # future record, because that IS h_{t+horizon}

        if not h_list:
            raise ValueError(f"No valid triples found for H={horizon}, contact_only={contact_only}")

        h = np.stack(h_list)
        a = np.stack(a_list)
        fh = np.stack(fh_list)
        mask = np.ones(len(h), dtype=bool)

    if contact_only and horizon == stored_H:
        h = h[mask]
        a = a[mask]
        fh = fh[mask]

    return h, a, fh


def _infonce_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    similarity: str = "cosine",
) -> torch.Tensor:
    """Contrastive loss that pulls predictions onto the real-feature manifold.

    For a batch of (predicted_i, real_i) pairs, encourages similarity between
    each predicted and its own target to exceed similarity to other targets
    in the batch. Directly penalizes off-manifold outputs.

    Two similarity modes:

    "cosine" — classic InfoNCE on L2-normalized vectors. Use when features
        have meaningful angular structure. *Degenerate* when all features
        point in nearly the same direction (e.g., Pi0.5's pooled hidden
        states cluster on a tight cone): all pairwise cosines ≈ 1.0,
        softmax saturates, and gradient vanishes.

    "l2" — InfoNCE on negative L2 distance. Use when feature magnitudes
        carry signal that L2-normalization would erase. The right choice
        for cone-shaped feature distributions where cosine is degenerate.

    Reference: van den Oord 2018 (cosine variant); the L2-variant is the
    metric-learning-with-NCE formulation common in self-supervised
    learning when features aren't unit-normalized (Hadsell et al. 2006
    contrastive loss in modern InfoNCE clothing).
    """
    if similarity == "cosine":
        p = torch.nn.functional.normalize(predicted, dim=-1)
        t = torch.nn.functional.normalize(target, dim=-1)
        logits = (p @ t.T) / max(temperature, 1e-6)
    elif similarity == "l2":
        # Negative L2 distance: closer pairs get higher logits.
        # cdist returns shape [B, B] of pairwise Euclidean distances.
        dists = torch.cdist(predicted, target, p=2)
        logits = -dists / max(temperature, 1e-6)
    else:
        raise ValueError(f"Unknown InfoNCE similarity mode: {similarity!r}")

    labels = torch.arange(len(predicted), device=predicted.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def _vicreg_loss(
    x: torch.Tensor,
    target_std: float | torch.Tensor = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
) -> torch.Tensor:
    """VICReg regularizer: push per-dim std toward target_std and decorrelate dims.

    Variance term prevents mean-regression collapse (LeWM's mechanism).
    Covariance term prevents dimensional redundancy.

    Reference: Bardes, Ponce, LeCun — VICReg (ICLR 2022); used here on the
    predictor's *output* distribution to attack predictor-side collapse.

    target_std can be either a scalar (classic VICReg target=1.0) or a
    per-dimension tensor of shape [D] so the regularizer pushes predicted
    features toward matching the *real* feature distribution's per-dim
    variance rather than an arbitrary unit-std target. Per-dim targeting
    is the correct move when the encoder outputs aren't normalized (our
    case: frozen Pi0.5 pooled VLM features).
    """
    B, D = x.shape
    std = torch.sqrt(x.var(dim=0) + 1e-4)                            # [D]

    if isinstance(target_std, torch.Tensor):
        target = target_std.to(x.device)
        if target.shape != std.shape:
            raise ValueError(
                f"target_std tensor shape {target.shape} != predictor output dims {std.shape}"
            )
    else:
        target = torch.full_like(std, float(target_std))

    L_var = torch.mean(torch.nn.functional.relu(target - std))

    # Covariance: penalize off-diagonal covariance
    x_centered = x - x.mean(dim=0, keepdim=True)
    cov = (x_centered.T @ x_centered) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    L_cov = (off_diag ** 2).sum() / D

    return variance_weight * L_var + covariance_weight * L_cov


def train_world_model(
    features: FeatureDataset,
    horizon: int,
    size: str = "medium",
    contact_only: bool = False,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    val_fraction: float = 0.15,
    patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # --- regularizers (default off — strict backward compat) ---
    use_infonce: bool = False,
    infonce_weight: float = 0.1,
    infonce_temperature: float = 0.1,
    infonce_similarity: str = "l2",
    use_vicreg: bool = False,
    vicreg_variance_weight: float = 1.0,
    vicreg_covariance_weight: float = 0.04,
    vicreg_target_std: float = 1.0,
    vicreg_match_real_std: bool = False,
) -> tuple[LatentWorldModel, dict]:
    """Train the world model and return the model + training metrics.

    Args:
        features: The extracted feature dataset.
        horizon: Action chunk horizon H.
        size: Model size ("small", "medium", "large").
        contact_only: Train on contact-phase timesteps only.
        num_epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_fraction: Fraction of data for validation.
        patience: Early stopping patience (epochs without val improvement).
        device: Training device.
        use_infonce: If True, add an InfoNCE contrastive term that pulls
            predicted features toward real features on the same manifold.
            Attacks manifold-drift failure mode (see diagnose_world_model.py).
        infonce_weight: Weight of the InfoNCE term relative to MSE.
        infonce_temperature: Softmax temperature for the InfoNCE logits.
        use_vicreg: If True, add VICReg regularization on the predictor's
            outputs. Attacks variance-collapse failure mode.
        vicreg_variance_weight: Weight of the variance-floor term.
        vicreg_covariance_weight: Weight of the off-diagonal covariance penalty.
        vicreg_target_std: Target per-dim std for the variance term.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    # Build dataset for this horizon
    h, a, fh = build_dataset_for_horizon(features, horizon, contact_only=contact_only)
    logger.info(f"Training data: {len(h)} triples, H={horizon}, contact_only={contact_only}")

    # Train/val split
    n = len(h)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    indices = np.random.RandomState(42).permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor(h[train_idx]), to_tensor(a[train_idx]), to_tensor(fh[train_idx]))
    val_ds = TensorDataset(to_tensor(h[val_idx]), to_tensor(a[val_idx]), to_tensor(fh[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Build model
    config = SIZE_CONFIGS[size]
    # Adjust hidden_dim and action_dim from data
    config = WorldModelConfig(
        hidden_dim=h.shape[1],
        action_dim=a.shape[2],
        max_horizon=features.horizon,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )
    model = LatentWorldModel(config).to(device)
    logger.info(f"Model: {size}, {model.param_count():,} params, d_model={config.d_model}, layers={config.n_layers}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    # Pre-compute the per-dim target std for VICReg — matches the real
    # feature distribution's per-dim variance rather than an arbitrary
    # unit-std target. Essential for frozen-encoder setups where the
    # features' native scale is whatever Pi0.5 learned.
    if use_vicreg and vicreg_match_real_std:
        real_std_per_dim = torch.tensor(
            fh[train_idx].std(axis=0), dtype=torch.float32
        ).to(device)
        logger.info(
            f"  VICReg target_std: per-dim (matched to real features). "
            f"mean={float(real_std_per_dim.mean()):.4f}, "
            f"median={float(real_std_per_dim.median()):.4f}, "
            f"max={float(real_std_per_dim.max()):.4f}"
        )
        vicreg_target: float | torch.Tensor = real_std_per_dim
    else:
        vicreg_target = vicreg_target_std

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Track component losses separately for diagnostics
    component_hist: dict[str, list[float]] = {"mse": [], "infonce": [], "vicreg": []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_infonce = 0.0
        epoch_vicreg = 0.0
        for batch_h, batch_a, batch_fh in train_loader:
            batch_h, batch_a, batch_fh = batch_h.to(device), batch_a.to(device), batch_fh.to(device)
            pred = model(batch_h, batch_a)

            L_mse = criterion(pred, batch_fh)
            loss = L_mse
            epoch_mse += L_mse.item() * len(batch_h)

            if use_infonce:
                L_nce = _infonce_loss(
                    pred, batch_fh, infonce_temperature, similarity=infonce_similarity,
                )
                loss = loss + infonce_weight * L_nce
                epoch_infonce += L_nce.item() * len(batch_h)

            if use_vicreg:
                L_vic = _vicreg_loss(
                    pred,
                    target_std=vicreg_target,
                    variance_weight=vicreg_variance_weight,
                    covariance_weight=vicreg_covariance_weight,
                )
                loss = loss + L_vic
                epoch_vicreg += L_vic.item() * len(batch_h)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_h)

        train_loss = epoch_loss / n_train
        train_losses.append(train_loss)
        component_hist["mse"].append(epoch_mse / n_train)
        component_hist["infonce"].append(epoch_infonce / n_train if use_infonce else 0.0)
        component_hist["vicreg"].append(epoch_vicreg / n_train if use_vicreg else 0.0)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_h, batch_a, batch_fh in val_loader:
                batch_h, batch_a, batch_fh = batch_h.to(device), batch_a.to(device), batch_fh.to(device)
                pred = model(batch_h, batch_a)
                loss = criterion(pred, batch_fh)
                val_loss += loss.item() * len(batch_h)
        val_loss = val_loss / n_val
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if epochs_without_improvement >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    # Compute no-change baseline (KS1)
    with torch.no_grad():
        all_h = to_tensor(h[val_idx]).to(device)
        all_fh = to_tensor(fh[val_idx]).to(device)
        no_change_loss = criterion(all_h, all_fh).item()

    metrics = {
        "train_loss_final": train_losses[-1],
        "val_loss_best": best_val_loss,
        "no_change_baseline": no_change_loss,
        "beats_no_change": best_val_loss < no_change_loss,
        "improvement_over_no_change": 1.0 - best_val_loss / no_change_loss if no_change_loss > 0 else 0.0,
        "epochs_trained": len(train_losses),
        "n_train": n_train,
        "n_val": n_val,
        "param_count": model.param_count(),
        "horizon": horizon,
        "size": size,
        "contact_only": contact_only,
        "use_infonce": use_infonce,
        "infonce_weight": infonce_weight if use_infonce else 0.0,
        "infonce_temperature": infonce_temperature if use_infonce else 0.0,
        "infonce_similarity": infonce_similarity if use_infonce else "",
        "use_vicreg": use_vicreg,
        "vicreg_variance_weight": vicreg_variance_weight if use_vicreg else 0.0,
        "vicreg_covariance_weight": vicreg_covariance_weight if use_vicreg else 0.0,
        "component_mse_final": component_hist["mse"][-1] if component_hist["mse"] else 0.0,
        "component_infonce_final": component_hist["infonce"][-1] if component_hist["infonce"] else 0.0,
        "component_vicreg_final": component_hist["vicreg"][-1] if component_hist["vicreg"] else 0.0,
    }

    logger.info(f"  Best val loss: {best_val_loss:.6f}")
    logger.info(f"  No-change baseline: {no_change_loss:.6f}")
    logger.info(f"  KS1 {'PASS' if metrics['beats_no_change'] else 'FAIL'}: "
                f"{'%.1f' % (metrics['improvement_over_no_change'] * 100)}% improvement over no-change")

    return model, metrics


def save_world_model(model: LatentWorldModel, metrics: dict, output_dir: str) -> None:
    """Save trained world model and metrics."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"H{metrics['horizon']}_{metrics['size']}"
    if metrics["contact_only"]:
        tag += "_contact"
    if metrics.get("use_infonce"):
        sim = metrics.get("infonce_similarity", "")
        # Cosine is the legacy default; tag L2 explicitly so checkpoints
        # don't collide with the older cosine variant.
        tag += "_nce" if sim in ("", "cosine") else f"_nce{sim}"
    if metrics.get("use_vicreg"):
        tag += "_vic"

    torch.save(model.state_dict(), output_dir / f"world_model_{tag}.pt")
    torch.save(model.config, output_dir / f"world_model_config_{tag}.pt")
    np.savez(output_dir / f"world_model_metrics_{tag}.npz", **{k: np.array(v) for k, v in metrics.items()})
    logger.info(f"Saved world model to {output_dir}/world_model_{tag}.*")
