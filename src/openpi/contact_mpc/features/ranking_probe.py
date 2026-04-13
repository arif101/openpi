"""Pairwise ranking probe: MLP trained with Bradley-Terry loss.

Unlike the linear probe (trained for classification, ranking is a side
effect), this probe is trained directly to rank hidden states by
temporal progress using pairwise comparisons. This produces a much
stronger proxy scorer for KS3 evaluation.

Architecture matches the planned value function (2-layer MLP) so this
is not throwaway work — it's an early version of the same component.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class RankingMLP(nn.Module):
    """2-layer MLP that scores hidden states for pairwise ranking."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a hidden state. Returns scalar per batch element."""
        return self.net(x).squeeze(-1)

    def predict_proba_numpy(self, x: np.ndarray) -> np.ndarray:
        """Score hidden states from numpy. Returns shape [N]."""
        self.eval()
        with torch.no_grad():
            scores = self(torch.tensor(x, dtype=torch.float32))
        return scores.numpy()


def build_ranking_pairs(
    hidden_states: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    n_pairs: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (h_better, h_worse) pairs from temporal progress within episodes.

    For each pair, h_better is from a later timestep in the same episode
    than h_worse. This is the training signal: later = closer to success
    in a successful demonstration.

    Args:
        hidden_states: Shape [N, hidden_dim].
        episode_ids: Shape [N].
        timesteps: Shape [N].
        n_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        Tuple of (h_better, h_worse), each shape [n_pairs, hidden_dim].
    """
    rng = np.random.RandomState(seed)

    # Group indices by episode
    ep_to_indices = {}
    for i in range(len(hidden_states)):
        ep = int(episode_ids[i])
        ep_to_indices.setdefault(ep, []).append(i)

    # Build pairs: sample episode, sample two timesteps, later one is "better"
    episodes = list(ep_to_indices.keys())
    h_better_list = []
    h_worse_list = []

    for _ in range(n_pairs):
        ep = rng.choice(episodes)
        indices = ep_to_indices[ep]
        if len(indices) < 2:
            continue
        i, j = rng.choice(len(indices), size=2, replace=False)
        idx_i, idx_j = indices[i], indices[j]
        if timesteps[idx_i] > timesteps[idx_j]:
            h_better_list.append(hidden_states[idx_i])
            h_worse_list.append(hidden_states[idx_j])
        else:
            h_better_list.append(hidden_states[idx_j])
            h_worse_list.append(hidden_states[idx_i])

    return np.stack(h_better_list), np.stack(h_worse_list)


def train_ranking_probe(
    hidden_states: np.ndarray,
    episode_ids: np.ndarray,
    timesteps: np.ndarray,
    hidden_dim: int = 256,
    n_pairs: int = 50000,
    n_val_pairs: int = 5000,
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[RankingMLP, float]:
    """Train a pairwise ranking probe with Bradley-Terry loss.

    Args:
        hidden_states: Feature matrix [N, hidden_dim].
        episode_ids: Episode IDs [N].
        timesteps: Timesteps [N].
        hidden_dim: MLP hidden layer size.
        n_pairs: Number of training pairs.
        n_val_pairs: Number of validation pairs.
        num_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Training device.

    Returns:
        Tuple of (trained_model, val_ranking_accuracy).
    """
    input_dim = hidden_states.shape[1]
    logger.info(f"Training on device: {device}")

    # Build train and val pairs
    h_better_train, h_worse_train = build_ranking_pairs(
        hidden_states, episode_ids, timesteps, n_pairs=n_pairs, seed=42
    )
    h_better_val, h_worse_val = build_ranking_pairs(
        hidden_states, episode_ids, timesteps, n_pairs=n_val_pairs, seed=99
    )

    logger.info(f"Training pairs: {len(h_better_train)}, Val pairs: {len(h_better_val)}")

    train_ds = TensorDataset(
        torch.tensor(h_better_train, dtype=torch.float32),
        torch.tensor(h_worse_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = RankingMLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    # Move val data to device once
    val_better_t = torch.tensor(h_better_val, dtype=torch.float32, device=device)
    val_worse_t = torch.tensor(h_worse_val, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for h_b, h_w in train_loader:
            h_b, h_w = h_b.to(device), h_w.to(device)
            score_better = model(h_b)
            score_worse = model(h_w)
            # Bradley-Terry loss: -log sigmoid(score_better - score_worse)
            loss = -torch.mean(torch.log(torch.sigmoid(score_better - score_worse) + 1e-8))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate: ranking accuracy on val pairs
        model.eval()
        with torch.no_grad():
            val_better_scores = model(val_better_t)
            val_worse_scores = model(val_worse_t)
            val_acc = (val_better_scores > val_worse_scores).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{num_epochs}: loss={epoch_loss/n_batches:.4f}, val_ranking_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    model.cpu().eval()
    logger.info(f"Best val ranking accuracy: {best_val_acc:.3f}")

    return model, best_val_acc


def evaluate_world_model_with_ranking_probe(
    world_model_path: str,
    features_path: str,
    ranking_probe: RankingMLP,
    horizon: int = 10,
    n_pairs: int = 1000,
) -> float:
    """Evaluate a world model's KS3 using the ranking probe instead of linear probe.

    Args:
        world_model_path: Path to saved world model .pt file.
        features_path: Path to features .npz file.
        ranking_probe: Trained RankingMLP.
        horizon: Action chunk horizon.
        n_pairs: Number of evaluation pairs.

    Returns:
        Ranking accuracy.
    """
    from openpi.contact_mpc.features.dataset import FeatureDataset
    from openpi.contact_mpc.world_model.architecture import LatentWorldModel
    from openpi.contact_mpc.world_model.train import build_dataset_for_horizon

    features = FeatureDataset.load(features_path)
    h, a, fh = build_dataset_for_horizon(features, horizon)

    ep_ids = features.episode_ids[:len(h)]
    timesteps = features.timesteps[:len(h)]
    ep_lens = {}
    for i in range(len(ep_ids)):
        ep_lens.setdefault(int(ep_ids[i]), 0)
        ep_lens[int(ep_ids[i])] = max(ep_lens[int(ep_ids[i])], int(timesteps[i]) + 1)

    early = [i for i in range(len(timesteps)) if timesteps[i] / ep_lens[int(ep_ids[i])] < 0.5]
    late = [i for i in range(len(timesteps)) if timesteps[i] / ep_lens[int(ep_ids[i])] > 0.75]

    rng = np.random.RandomState(42)
    n_pairs = min(n_pairs, len(early), len(late))
    ei = rng.choice(early, n_pairs, replace=True)
    li = rng.choice(late, n_pairs, replace=True)

    # Load world model
    config = torch.load(world_model_path.replace(".pt", "").replace("world_model_", "world_model_config_") + ".pt",
                        weights_only=False)
    wm = LatentWorldModel(config)
    wm.load_state_dict(torch.load(world_model_path, weights_only=True))
    wm.eval()

    # Predict future states
    with torch.no_grad():
        pred_early = wm(
            torch.tensor(h[ei], dtype=torch.float32),
            torch.tensor(a[ei], dtype=torch.float32),
        )
        pred_late = wm(
            torch.tensor(h[li], dtype=torch.float32),
            torch.tensor(a[li], dtype=torch.float32),
        )

    # Score with ranking probe
    ranking_probe.eval()
    with torch.no_grad():
        early_scores = ranking_probe(pred_early)
        late_scores = ranking_probe(pred_late)
        acc = (late_scores > early_scores).float().mean().item()

    logger.info(f"KS3 (ranking probe): {acc:.3f} (threshold=0.70)")
    return acc
