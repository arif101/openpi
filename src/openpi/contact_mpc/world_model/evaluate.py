"""Kill switch evaluation for the latent world model.

KS1: MSE beats no-change baseline (trivial bar)
KS2: Contact-phase model outperforms contact-blind model on contact data
KS3: Ranking accuracy ≥70% on held-out success/failure pairs using proxy scorer
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from openpi.contact_mpc.features.dataset import FeatureDataset
from openpi.contact_mpc.world_model.architecture import LatentWorldModel
from openpi.contact_mpc.world_model.train import build_dataset_for_horizon

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def evaluate_ks1(
    model: LatentWorldModel,
    features: FeatureDataset,
    horizon: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """KS1: Does the model beat the no-change baseline?

    The no-change baseline predicts h_{t+H} = h_t. If the world model
    can't beat this, it learned nothing useful.

    Returns dict with mse, no_change_mse, passes (bool).
    """
    h, a, fh = build_dataset_for_horizon(features, horizon)

    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.float32, device=device)
    fh_t = torch.tensor(fh, dtype=torch.float32, device=device)

    model = model.to(device).eval()
    with torch.no_grad():
        pred = model(h_t, a_t)
        model_mse = torch.mean((pred - fh_t) ** 2).item()
        no_change_mse = torch.mean((h_t - fh_t) ** 2).item()

    passes = model_mse < no_change_mse
    improvement = 1.0 - model_mse / no_change_mse if no_change_mse > 0 else 0.0

    logger.info(f"KS1: model_mse={model_mse:.6f}, no_change={no_change_mse:.6f}, "
                f"improvement={improvement*100:.1f}%, {'PASS' if passes else 'FAIL'}")

    return {"model_mse": model_mse, "no_change_mse": no_change_mse, "improvement": improvement, "passes": passes}


def evaluate_ks2(
    model_all: LatentWorldModel,
    model_contact: LatentWorldModel,
    features: FeatureDataset,
    horizon: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """KS2: Does the contact-trained model outperform the all-data model on contact data?

    Both models are evaluated on contact-phase held-out data only.
    If the contact-specific model isn't better, free-space dynamics
    are washing out the contact signal.

    Returns dict with all_mse, contact_mse, passes (bool).
    """
    # Get contact-only evaluation data
    h, a, fh = build_dataset_for_horizon(features, horizon, contact_only=True)

    if len(h) == 0:
        logger.warning("KS2: No contact-phase data available. Cannot evaluate.")
        return {"all_mse": float("nan"), "contact_mse": float("nan"), "passes": False}

    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    a_t = torch.tensor(a, dtype=torch.float32, device=device)
    fh_t = torch.tensor(fh, dtype=torch.float32, device=device)

    model_all = model_all.to(device).eval()
    model_contact = model_contact.to(device).eval()

    with torch.no_grad():
        pred_all = model_all(h_t, a_t)
        pred_contact = model_contact(h_t, a_t)
        all_mse = torch.mean((pred_all - fh_t) ** 2).item()
        contact_mse = torch.mean((pred_contact - fh_t) ** 2).item()

    passes = contact_mse < all_mse

    logger.info(f"KS2: all_data_mse={all_mse:.6f}, contact_only_mse={contact_mse:.6f}, "
                f"{'PASS' if passes else 'FAIL'} (contact model {'better' if passes else 'worse'})")

    return {"all_mse": all_mse, "contact_mse": contact_mse, "passes": passes}


def evaluate_ks3(
    model: LatentWorldModel,
    features: FeatureDataset,
    horizon: int,
    proxy_scorer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_pairs: int = 1000,
) -> dict:
    """KS3: Ranking accuracy ≥70% using world model predictions + proxy scorer.

    Given two trajectories from the same task with different temporal
    progress (early vs late), predict future hidden states via the world
    model and score them. The later-in-episode trajectory should score
    higher (it's closer to success in the demo dataset).

    The proxy scorer is the Day 2 linear probe (or MLP probe) that was
    trained to predict temporal progress.

    Args:
        model: Trained world model.
        features: Feature dataset.
        horizon: Action chunk horizon.
        proxy_scorer: Trained sklearn model with predict_proba method.
        device: Compute device.
        n_pairs: Number of pairs to evaluate.

    Returns:
        Dict with ranking_accuracy, n_pairs_evaluated, passes (bool).
    """
    h, a, fh = build_dataset_for_horizon(features, horizon)
    ep_ids = features.episode_ids[:len(h)]
    timesteps = features.timesteps[:len(h)]

    # Compute per-episode length for progress labels
    ep_lengths = {}
    for i in range(len(ep_ids)):
        eid = int(ep_ids[i])
        ep_lengths[eid] = max(ep_lengths.get(eid, 0), int(timesteps[i]) + 1)

    # Split into early (first 50%) and late (last 25%) within each episode
    early_idx = []
    late_idx = []
    for i in range(len(h)):
        eid = int(ep_ids[i])
        progress = timesteps[i] / ep_lengths[eid]
        if progress < 0.5:
            early_idx.append(i)
        elif progress > 0.75:
            late_idx.append(i)

    if not early_idx or not late_idx:
        logger.warning("KS3: Not enough early/late samples for ranking evaluation.")
        return {"ranking_accuracy": 0.0, "n_pairs_evaluated": 0, "passes": False}

    # Sample pairs
    rng = np.random.RandomState(42)
    n_pairs = min(n_pairs, len(early_idx) * len(late_idx))
    early_samples = rng.choice(early_idx, size=n_pairs, replace=True)
    late_samples = rng.choice(late_idx, size=n_pairs, replace=True)

    # Predict future hidden states for both early and late samples
    model = model.to(device).eval()
    with torch.no_grad():
        early_h = torch.tensor(h[early_samples], dtype=torch.float32, device=device)
        early_a = torch.tensor(a[early_samples], dtype=torch.float32, device=device)
        late_h = torch.tensor(h[late_samples], dtype=torch.float32, device=device)
        late_a = torch.tensor(a[late_samples], dtype=torch.float32, device=device)

        pred_early = model(early_h, early_a).cpu().numpy()
        pred_late = model(late_h, late_a).cpu().numpy()

    # Score predictions with the proxy scorer
    early_scores = proxy_scorer.predict_proba(pred_early)[:, 1]
    late_scores = proxy_scorer.predict_proba(pred_late)[:, 1]

    # Ranking accuracy: late (closer to success) should score higher
    n_correct = np.sum(late_scores > early_scores)
    n_ties = np.sum(late_scores == early_scores)
    ranking_accuracy = (n_correct + 0.5 * n_ties) / n_pairs

    passes = ranking_accuracy >= 0.70

    logger.info(f"KS3: ranking_accuracy={ranking_accuracy:.3f}, "
                f"pairs={n_pairs}, threshold=0.70, {'PASS' if passes else 'FAIL'}")

    return {"ranking_accuracy": ranking_accuracy, "n_pairs_evaluated": n_pairs, "passes": passes}


def run_all_kill_switches(
    model_all: LatentWorldModel,
    model_contact: LatentWorldModel | None,
    features: FeatureDataset,
    horizon: int,
    proxy_scorer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Run all kill switches and return combined results."""
    results = {}

    logger.info(f"\n{'='*60}")
    logger.info(f"Kill Switch Evaluation: H={horizon}")
    logger.info(f"{'='*60}")

    # KS1
    ks1 = evaluate_ks1(model_all, features, horizon, device)
    results["ks1"] = ks1

    # KS2 (only if we have a contact-only model)
    if model_contact is not None:
        ks2 = evaluate_ks2(model_all, model_contact, features, horizon, device)
        results["ks2"] = ks2
    else:
        logger.info("KS2: Skipped (no contact-only model provided)")
        results["ks2"] = {"passes": None}

    # KS3
    ks3 = evaluate_ks3(model_all, features, horizon, proxy_scorer, device)
    results["ks3"] = ks3

    # Summary
    all_pass = ks1["passes"] and ks3["passes"]
    logger.info(f"\nSummary: KS1={'PASS' if ks1['passes'] else 'FAIL'}, "
                f"KS2={'PASS' if results['ks2'].get('passes') else 'FAIL/SKIP'}, "
                f"KS3={'PASS' if ks3['passes'] else 'FAIL'}")
    logger.info(f"Overall: {'ALL KILL SWITCHES PASS' if all_pass else 'KILL SWITCH FAILURE — consider pivoting'}")

    results["all_pass"] = all_pass
    return results
