"""Train the pairwise value function with Bradley-Terry loss."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openpi.contact_mpc.value_function.architecture import PairwiseValueFunction

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def train_value_function(
    h_success: np.ndarray,
    h_failure: np.ndarray,
    val_success: np.ndarray | None = None,
    val_failure: np.ndarray | None = None,
    hidden_dim: int = 256,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[PairwiseValueFunction, dict]:
    """Train value function with Bradley-Terry loss.

    Args:
        h_success: [N, hidden_dim] success hidden states.
        h_failure: [N, hidden_dim] failure hidden states (paired).
        val_success: Optional validation success states.
        val_failure: Optional validation failure states.
        hidden_dim: MLP hidden layer size.
        num_epochs: Maximum epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        device: Training device.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    input_dim = h_success.shape[1]
    logger.info(f"Training on device: {device}")
    logger.info(f"Training pairs: {len(h_success)}, input_dim: {input_dim}")

    # Build data loaders
    train_ds = TensorDataset(
        torch.tensor(h_success, dtype=torch.float32),
        torch.tensor(h_failure, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation data
    if val_success is not None and val_failure is not None:
        val_s = torch.tensor(val_success, dtype=torch.float32, device=device)
        val_f = torch.tensor(val_failure, dtype=torch.float32, device=device)
        has_val = True
    else:
        # Use last 15% of training data as validation
        n_val = max(1, int(len(h_success) * 0.15))
        val_s = torch.tensor(h_success[-n_val:], dtype=torch.float32, device=device)
        val_f = torch.tensor(h_failure[-n_val:], dtype=torch.float32, device=device)
        has_val = True

    # Build model
    model = PairwiseValueFunction(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f"Model: {model.param_count():,} params")

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0
    train_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_s, batch_f in train_loader:
            batch_s, batch_f = batch_s.to(device), batch_f.to(device)
            score_s = model(batch_s)
            score_f = model(batch_f)
            # Bradley-Terry loss
            loss = -torch.mean(torch.log(torch.sigmoid(score_s - score_f) + 1e-8))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_score_s = model(val_s)
            val_score_f = model(val_f)
            val_acc = (val_score_s > val_score_f).float().mean().item()
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={train_loss:.4f}, val_acc={val_acc:.3f}", flush=True)

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.cpu().eval()

    metrics = {
        "val_ranking_accuracy": best_val_acc,
        "train_loss_final": train_losses[-1],
        "epochs_trained": len(train_losses),
        "param_count": model.param_count(),
        "passes_70_threshold": best_val_acc >= 0.70,
    }

    print(f"  Best val ranking accuracy: {best_val_acc:.3f}", flush=True)
    print(f"  KS threshold (≥0.70): {'PASS' if metrics['passes_70_threshold'] else 'FAIL'}", flush=True)

    return model, metrics
