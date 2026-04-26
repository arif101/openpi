"""Value functions for ranking action chunks during MCTS / MPC inference.

Two flavors:

- ``PairwiseValueFunction``  — V(h). 2-layer MLP scoring a hidden state.
  Trained with Bradley-Terry loss on (h_success, h_failure) pairs. Cannot
  differentiate candidates whose action chunks lead to similar latent states.

- ``ActionConditionalValueFunction`` — Q(h, a). Scores (hidden_state,
  action_chunk) jointly. Differentiates candidates by the action that led
  to a leaf, even when the post-action latent is similar. Naturally fits
  MCTS leaf evaluation where each child node has a distinct action chunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class PairwiseValueFunction(nn.Module):
    """2-layer MLP that scores hidden states (V(h))."""

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


class ActionConditionalValueFunction(nn.Module):
    """Q(h, a) — scores (hidden_state, action_chunk) jointly.

    Architecture:
      - Action encoder: flattens [H, action_dim] → MLP → action_emb [action_emb_dim]
      - Concatenates [hidden_state, action_emb] → 2-layer scoring MLP → scalar

    The action encoder lets the model learn action-relevant features rather
    than overfitting to raw chunk numerics. Dropout + weight decay reduce
    the task-identity memorization that V(h) suffered from on LIBERO-90.
    """

    def __init__(
        self,
        hidden_state_dim: int = 2048,
        action_chunk_horizon: int = 10,
        action_dim: int = 7,
        action_emb_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.action_chunk_horizon = action_chunk_horizon
        self.action_dim = action_dim
        self.action_emb_dim = action_emb_dim

        flat_action = action_chunk_horizon * action_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(flat_action, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_emb_dim),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_state_dim + action_emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.action_encoder) + list(self.scorer):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_action(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Encode [B, H, action_dim] → [B, action_emb_dim]."""
        b = action_chunk.shape[0]
        flat = action_chunk.reshape(b, -1)
        return self.action_encoder(flat)

    def forward(
        self,
        hidden_state: torch.Tensor,    # [B, hidden_state_dim]
        action_chunk: torch.Tensor,    # [B, H, action_dim]
    ) -> torch.Tensor:
        """Score (h, a) pairs. Returns [B] scalar scores."""
        action_emb = self.encode_action(action_chunk)
        x = torch.cat([hidden_state, action_emb], dim=-1)
        return self.scorer(x).squeeze(-1)

    def score_numpy(
        self,
        hidden_state: np.ndarray,
        action_chunk: np.ndarray,
    ) -> np.ndarray:
        """Score (h, a) pairs from numpy arrays. Adds batch dim if needed."""
        self.eval()
        h = torch.tensor(hidden_state, dtype=torch.float32)
        a = torch.tensor(action_chunk, dtype=torch.float32)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if a.ndim == 2:
            a = a.unsqueeze(0)
        with torch.no_grad():
            return self(h, a).numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
