"""Flow-matching DPO loss for Pi0.5 LoRA training.

Adapts Wallace et al. 2023 ("Diffusion Model Alignment via DPO") to flow
matching. Pi0.5 generates action chunks by integrating an ODE whose
vector field v_θ(x_t, t, obs) is trained to predict (x - ε) — the
flow-matching regression target.

For a preference pair (obs_w, x_w) vs (obs_l, x_l), we sample one
timestep t and one noise vector ε per trajectory and compute:

    L_FM(θ; x, obs) = || v_θ((1-t)·x + t·ε, t, obs) - (x - ε) ||^2

The DPO loss is:

    L_DPO(θ) = -log σ( -β · [L_FM(θ; x_w) - L_FM(θ_ref; x_w)
                             - L_FM(θ; x_l) + L_FM(θ_ref; x_l)] )

Intuition: reduce θ's flow-matching error on winners relative to the
reference and raise it on losers. Gradient direction is opposite to
the L_FM gradient sign, so this is NOT equivalent to plain SFT.

This module is framework-agnostic for the mathematical kernel
(`flow_matching_dpo_loss` works on any tensor library with broadcasting
and a `sigmoid` / `logsigmoid` primitive). Plumbing for Pi0.5's JAX
forward pass lives in `trainer.py`.
"""

from __future__ import annotations

import numpy as np


def flow_matching_regression_loss(
    v_pred: np.ndarray,
    v_target: np.ndarray,
    *,
    reduce_axes: tuple[int, ...] = (-2, -1),
) -> np.ndarray:
    """Per-sample flow-matching regression loss.

    Args:
        v_pred: [..., H, action_dim] predicted vector field values.
        v_target: [..., H, action_dim] target (x - ε).
        reduce_axes: axes to mean over (default: action chunk + action dim,
            keeps the leading batch axis).

    Returns:
        Per-sample scalar loss of shape [...], with reduce_axes removed.
    """
    return np.mean((v_pred - v_target) ** 2, axis=reduce_axes)


def dpo_logits(
    l_fm_theta_win: np.ndarray,
    l_fm_ref_win: np.ndarray,
    l_fm_theta_lose: np.ndarray,
    l_fm_ref_lose: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute the DPO inner logit for a batch of pairs.

    Logit = -β · [L_FM(θ; x_w) - L_FM(ref; x_w)
                  - L_FM(θ; x_l) + L_FM(ref; x_l)]

    Following Wallace et al. 2023's diffusion-DPO derivation, the negative
    sign on the leading β folds the direction correctly so that the
    gradient reduces L_FM on winners and increases it on losers.

    Shapes: all inputs are [batch] per-sample losses. Returns [batch].
    """
    win_gap = l_fm_theta_win - l_fm_ref_win
    lose_gap = l_fm_theta_lose - l_fm_ref_lose
    return -beta * (win_gap - lose_gap)


def dpo_loss_from_fm(
    l_fm_theta_win: np.ndarray,
    l_fm_ref_win: np.ndarray,
    l_fm_theta_lose: np.ndarray,
    l_fm_ref_lose: np.ndarray,
    *,
    beta: float = 0.1,
) -> dict[str, np.ndarray]:
    """End-to-end DPO loss from four per-sample flow-matching losses.

    Returns a dict with:
        loss           — scalar mean DPO loss (for the optimizer)
        logits         — per-sample inner logit (diagnostic)
        accuracy       — fraction of pairs where θ prefers winner over loser
        implicit_reward_win  — β · (L_ref - L_θ) on winners (should rise)
        implicit_reward_lose — β · (L_ref - L_θ) on losers (should fall)
    """
    logits = dpo_logits(
        l_fm_theta_win, l_fm_ref_win, l_fm_theta_lose, l_fm_ref_lose, beta,
    )
    # log σ(x) computed stably: -log(1 + exp(-x))
    # For numpy path we use scipy's logsigmoid if available; otherwise an
    # inline stable form.
    try:
        from scipy.special import log_expit
        log_sig = log_expit(logits)
    except ImportError:  # pragma: no cover
        # Stable log σ(x) = min(x, 0) - log(1 + exp(-|x|))
        log_sig = np.minimum(logits, 0.0) - np.log1p(np.exp(-np.abs(logits)))

    loss = -np.mean(log_sig)

    # Diagnostics (NOT used in optimization)
    implicit_reward_win = beta * (l_fm_ref_win - l_fm_theta_win)
    implicit_reward_lose = beta * (l_fm_ref_lose - l_fm_theta_lose)
    accuracy = float(np.mean(logits > 0.0))

    return {
        "loss": loss,
        "logits": logits,
        "accuracy": np.asarray(accuracy),
        "implicit_reward_win": implicit_reward_win,
        "implicit_reward_lose": implicit_reward_lose,
    }


def sample_flow_matching_noise(
    action_chunk: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (t, ε, x_t) for flow-matching training.

    Args:
        action_chunk: [batch, H, action_dim] target action.
        rng: numpy RNG.

    Returns:
        t:   [batch, 1, 1] — sampled timestep in (0, 1).
        eps: [batch, H, action_dim] — standard normal noise.
        x_t: [batch, H, action_dim] — noised action = (1-t)·x + t·ε.

    The target velocity v_target = x - ε is computable by the caller.
    """
    batch = action_chunk.shape[0]
    t = rng.uniform(low=0.0, high=1.0, size=(batch, 1, 1)).astype(np.float32)
    eps = rng.standard_normal(size=action_chunk.shape).astype(np.float32)
    x_t = (1.0 - t) * action_chunk + t * eps
    return t, eps, x_t
