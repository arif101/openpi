"""Thin wrapper around Pi0.extract_vlm_features for batch feature extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

from openpi.models import model as _model

logger = logging.getLogger(__name__)


def extract_features_from_observation(
    model: "Pi0",
    observation: _model.Observation,
) -> np.ndarray:
    """Extract pooled VLM hidden states from a batch of observations.

    Args:
        model: A frozen Pi0/Pi0.5 model instance.
        observation: Batched observation (images, state, tokenized prompt).

    Returns:
        Pooled hidden states as numpy array of shape [batch_size, hidden_dim].
    """
    pooled = model.extract_vlm_features(observation)
    return np.asarray(pooled)


def extract_features_from_dict(
    model: "Pi0",
    data: dict,
) -> np.ndarray:
    """Extract pooled VLM hidden states from a raw data dictionary.

    Convenience wrapper that converts a data dict to an Observation first.
    The data dict should have the same format as produced by the data transforms
    (images under "image", masks under "image_mask", state under "state", etc.).

    Args:
        model: A frozen Pi0/Pi0.5 model instance.
        data: Raw data dictionary with image, image_mask, state, and
            optionally tokenized_prompt / tokenized_prompt_mask.

    Returns:
        Pooled hidden states as numpy array of shape [batch_size, hidden_dim].
    """
    observation = _model.Observation.from_dict(data)
    return extract_features_from_observation(model, observation)


def get_hidden_dim(model: "Pi0") -> int:
    """Return the VLM hidden state dimension for a given model.

    Useful for constructing downstream architectures (world model, value
    function) that need to know the input dimension.
    """
    # The PaliGemma variant determines the hidden dim. For gemma_2b it's 2048.
    # We read it from the model's action_in_proj input features since the action
    # expert config's width is what embed_prefix outputs are projected to by the LLM.
    # A cleaner way: just read the LLM config. But this works without touching internals.
    return model.action_in_proj.out_features


def pool_with_mask(
    hidden_states: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Mean-pool hidden states over valid (non-padding) tokens.

    Args:
        hidden_states: Shape [batch, seq_len, hidden_dim].
        mask: Boolean mask of shape [batch, seq_len]. True for valid tokens.

    Returns:
        Pooled states of shape [batch, hidden_dim].
    """
    mask_expanded = mask[:, :, None].astype(hidden_states.dtype)
    return jnp.sum(hidden_states * mask_expanded, axis=1) / jnp.maximum(
        jnp.sum(mask_expanded, axis=1), 1.0
    )
