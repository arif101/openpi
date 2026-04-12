"""Latent world model: predicts future VLM hidden states from current state + action chunk.

Architecture: small transformer with per-timestep action tokenization.
Input: (h_t [2048-dim], action_chunk [H, 7]) → Output: ĥ_{t+H} [2048-dim]

The h_t context token occupies position 0. Each of the H action vectors
becomes one token at positions 1..H. Learned absolute positional embeddings
ensure the model can distinguish temporal order within the chunk.

Three size configs are provided for the sweep: small (~1M), medium (~5M),
large (~20M). Selection rule: smallest that passes KS3 at ≥70%.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for the latent world model."""

    hidden_dim: int = 2048  # VLM hidden state dimension (input/output)
    action_dim: int = 7  # per-timestep action dimension
    max_horizon: int = 10  # maximum H for positional embeddings
    d_model: int = 256  # transformer internal dimension
    n_heads: int = 4  # number of attention heads
    n_layers: int = 4  # number of transformer layers
    dropout: float = 0.1

    @property
    def param_count_estimate(self) -> int:
        """Rough parameter count estimate."""
        # Input projections: hidden_dim * d_model + action_dim * d_model
        proj = self.hidden_dim * self.d_model + self.action_dim * self.d_model
        # Transformer layers: ~4 * d_model^2 per layer (QKV + FFN)
        transformer = self.n_layers * 4 * self.d_model ** 2
        # Output projection: d_model * hidden_dim
        out = self.d_model * self.hidden_dim
        return proj + transformer + out


# Pre-defined size configs for the sweep
SMALL_CONFIG = WorldModelConfig(d_model=128, n_heads=4, n_layers=2)   # ~1M params
MEDIUM_CONFIG = WorldModelConfig(d_model=256, n_heads=4, n_layers=4)  # ~5M params
LARGE_CONFIG = WorldModelConfig(d_model=512, n_heads=8, n_layers=8)   # ~20M params

SIZE_CONFIGS = {
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,
}


class LatentWorldModel(nn.Module):
    """Predicts future VLM hidden states from (h_t, action_chunk).

    Architecture:
        1. Project h_t (2048-dim) → context token (d_model-dim)
        2. Project each action (7-dim) → action token (d_model-dim)
        3. Add learned positional embeddings (pos 0 = context, pos 1..H = actions)
        4. Run through transformer encoder
        5. Take the context token's output and project back to 2048-dim
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.hidden_proj = nn.Linear(config.hidden_dim, config.d_model)
        self.action_proj = nn.Linear(config.action_dim, config.d_model)

        # Learned positional embeddings: pos 0 = context token, pos 1..max_H = action tokens
        self.pos_embed = nn.Embedding(config.max_horizon + 1, config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Output projection: context token → predicted hidden state
        self.output_proj = nn.Linear(config.d_model, config.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for projections."""
        for m in [self.hidden_proj, self.action_proj, self.output_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future hidden state.

        Args:
            hidden_state: Current VLM hidden state, shape [batch, hidden_dim].
            action_chunk: Action sequence, shape [batch, H, action_dim].

        Returns:
            Predicted future hidden state, shape [batch, hidden_dim].
        """
        batch_size, H, _ = action_chunk.shape

        # Project inputs to d_model
        context_token = self.hidden_proj(hidden_state).unsqueeze(1)  # [B, 1, d_model]
        action_tokens = self.action_proj(action_chunk)  # [B, H, d_model]

        # Add positional embeddings
        context_pos = self.pos_embed(torch.zeros(batch_size, 1, dtype=torch.long, device=hidden_state.device))
        action_positions = torch.arange(1, H + 1, device=hidden_state.device).unsqueeze(0).expand(batch_size, -1)
        action_pos = self.pos_embed(action_positions)

        context_token = context_token + context_pos
        action_tokens = action_tokens + action_pos

        # Concatenate: [context, action_1, action_2, ..., action_H]
        tokens = torch.cat([context_token, action_tokens], dim=1)  # [B, 1+H, d_model]

        # Transformer forward
        output = self.transformer(tokens)  # [B, 1+H, d_model]

        # Take context token output (position 0) and project to hidden_dim
        predicted = self.output_proj(output[:, 0, :])  # [B, hidden_dim]

        return predicted

    def param_count(self) -> int:
        """Actual parameter count."""
        return sum(p.numel() for p in self.parameters())
