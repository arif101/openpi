"""Pairwise value function: 2-layer MLP trained with Bradley-Terry loss.

Scores hidden states for success probability. Used to rank K=8
candidate action chunks during MPC inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class PairwiseValueFunction(nn.Module):
    """2-layer MLP that scores hidden states."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score hidden states. Returns scalar per batch element."""
        return self.net(x).squeeze(-1)

    def score_numpy(self, x: np.ndarray) -> np.ndarray:
        """Score hidden states from numpy array."""
        self.eval()
        with torch.no_grad():
            scores = self(torch.tensor(x, dtype=torch.float32))
        return scores.numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
