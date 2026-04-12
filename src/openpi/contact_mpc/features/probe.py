"""Probes on VLM hidden states to test representation quality.

The earliest possible test of the central hypothesis: if a classifier
on frozen hidden states can predict LIBERO-90 success/progress
substantially above chance, the representation contains success-relevant
information and the search hypothesis is viable.

We run two probes:
1. Linear probe (logistic regression) — if this passes, the signal is
   linearly separable, which is the strongest possible result.
2. MLP probe (2-layer, same architecture as the real value function) —
   if the linear probe fails but this passes, the signal exists but
   needs nonlinear decoding. Still viable for our value function.

If both fail, the representation likely doesn't encode the information
we need, and the search hypothesis is in trouble.

The best probe also serves as the proxy scorer for F_demo's KS3
evaluation (ranking accuracy) in Week 1, before the real value function
is trained.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def train_linear_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> tuple[LogisticRegression, float]:
    """Train a linear probe with cross-validation and return the best model.

    Args:
        hidden_states: Feature matrix of shape [N, hidden_dim].
        labels: Binary labels of shape [N] (1 = success, 0 = failure).
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (best_model, mean_cv_accuracy).
    """
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"Need both positive and negative examples. Got labels: {np.unique(labels)}"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    best_acc = 0.0
    best_model = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(hidden_states, labels)):
        X_train, X_val = hidden_states[train_idx], hidden_states[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model

        logger.info(f"  Fold {fold_idx + 1}/{n_folds}: accuracy = {acc:.3f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    logger.info(f"Linear probe CV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

    return best_model, mean_acc


def train_mlp_probe(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> tuple[MLPClassifier, float]:
    """Train a 2-layer MLP probe with cross-validation.

    Same as train_linear_probe but uses a shallow MLP (256 hidden units,
    ReLU) instead of logistic regression. This is the fallback if the
    linear probe fails — if the MLP passes, the signal exists but needs
    nonlinear decoding, which is fine because our real value function is
    also a 2-layer MLP.

    Args:
        hidden_states: Feature matrix of shape [N, hidden_dim].
        labels: Binary labels of shape [N] (1 = success, 0 = failure).
        n_folds: Number of cross-validation folds.

    Returns:
        Tuple of (best_model, mean_cv_accuracy).
    """
    if len(np.unique(labels)) < 2:
        raise ValueError(
            f"Need both positive and negative examples. Got labels: {np.unique(labels)}"
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    best_acc = 0.0
    best_model = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(hidden_states, labels)):
        X_train, X_val = hidden_states[train_idx], hidden_states[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        fold_accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model

        logger.info(f"  Fold {fold_idx + 1}/{n_folds}: MLP accuracy = {acc:.3f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    logger.info(f"MLP probe CV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

    return best_model, mean_acc


def score_hidden_states(
    model: LogisticRegression,
    hidden_states: np.ndarray,
) -> np.ndarray:
    """Score hidden states using a trained linear probe.

    Returns the predicted probability of success for each hidden state.
    Used as the proxy scorer for KS3 ranking accuracy evaluation.

    Args:
        model: Trained logistic regression probe.
        hidden_states: Feature matrix of shape [N, hidden_dim].

    Returns:
        Success probabilities of shape [N].
    """
    return model.predict_proba(hidden_states)[:, 1]


def evaluate_ranking_accuracy(
    model: LogisticRegression,
    success_hidden_states: np.ndarray,
    failure_hidden_states: np.ndarray,
) -> float:
    """Evaluate pairwise ranking accuracy using the linear probe.

    For each (success, failure) pair, check if the probe scores the
    success state higher. This is the proxy KS3 metric for F_demo
    evaluation before the real value function is trained.

    Args:
        model: Trained logistic regression probe.
        success_hidden_states: Shape [N_success, hidden_dim].
        failure_hidden_states: Shape [N_failure, hidden_dim].

    Returns:
        Fraction of pairs where success scores higher than failure.
    """
    success_scores = score_hidden_states(model, success_hidden_states)
    failure_scores = score_hidden_states(model, failure_hidden_states)

    # Compare all pairs
    n_correct = 0
    n_total = 0
    for s_score in success_scores:
        for f_score in failure_scores:
            n_total += 1
            if s_score > f_score:
                n_correct += 1
            elif s_score == f_score:
                n_correct += 0.5  # tie = half credit

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    logger.info(
        f"Ranking accuracy: {accuracy:.3f} "
        f"({n_correct:.0f}/{n_total} pairs, "
        f"{len(success_scores)} successes vs {len(failure_scores)} failures)"
    )
    return accuracy
