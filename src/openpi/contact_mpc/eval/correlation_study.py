"""Imagined-vs-real correlation study for LoRA improvement candidates.

The demo's defensible technical claim is that the latent world model's
imagined score for a LoRA candidate correlates with the candidate's real
LIBERO success rate. This module computes Pearson r + a 95% bootstrap
confidence interval, and produces a scatter plot.

Kill-criterion bands from the plan:
  - r >= 0.5:        strong; ship the full demo with the validation claim
  - 0.3 <= r < 0.5:  medium; ship with "early signal, pilot pending" caveat
  - r < 0.3:         weak; strip the correlation beat from the demo

Pure numpy/scipy; no torch or JAX.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any, Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

Strength = Literal["strong", "medium", "weak"]


@dataclasses.dataclass
class CorrelationResult:
    """Output of a single imagined-vs-real correlation run."""

    pearson_r: float
    pearson_p_value: float
    spearman_r: float
    spearman_p_value: float
    n_pairs: int
    ci_low: float
    ci_high: float
    ci_level: float
    bootstrap_mean_r: float
    strength: Strength
    imagined: list[float]
    real: list[float]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def classify_correlation_strength(r: float) -> Strength:
    """Map a Pearson r to one of the three demo-outcome bands from the plan."""
    if r >= 0.5:
        return "strong"
    if r >= 0.3:
        return "medium"
    return "weak"


def compute_correlation(
    imagined: np.ndarray,
    real: np.ndarray,
    *,
    labels: list[str] | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> CorrelationResult:
    """Pearson r + Spearman rho with a bootstrap CI on Pearson r.

    Args:
        imagined: [N] imagined-improvement scores (from offline evaluator).
        real: [N] real success rates for the same candidate/task pairs.
        labels: [N] human-readable labels (candidate_name or "candidate@task").
        n_bootstrap: resamples for CI.
        ci_level: confidence level, e.g. 0.95.
        seed: RNG seed.

    Returns:
        CorrelationResult.

    Raises:
        ValueError if arrays are < 3 long or shape-mismatched.
    """
    imagined = np.asarray(imagined, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)

    if imagined.shape != real.shape:
        raise ValueError(
            f"Shape mismatch: imagined {imagined.shape} vs real {real.shape}"
        )
    if imagined.ndim != 1 or imagined.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 pairs for meaningful correlation, got {imagined.shape[0]}"
        )

    n = imagined.shape[0]
    if labels is None:
        labels = [f"pair_{i}" for i in range(n)]

    # Guard against degenerate inputs (all identical values)
    if np.std(imagined) < 1e-12 or np.std(real) < 1e-12:
        logger.warning("One of the input series has near-zero variance; r is undefined.")
        return CorrelationResult(
            pearson_r=float("nan"), pearson_p_value=float("nan"),
            spearman_r=float("nan"), spearman_p_value=float("nan"),
            n_pairs=n, ci_low=float("nan"), ci_high=float("nan"),
            ci_level=ci_level, bootstrap_mean_r=float("nan"),
            strength="weak",
            imagined=imagined.tolist(), real=real.tolist(), labels=labels,
        )

    pearson_r, pearson_p = stats.pearsonr(imagined, real)
    spearman_r, spearman_p = stats.spearmanr(imagined, real)

    # Bootstrap CI on Pearson r
    rng = np.random.default_rng(seed)
    boot_rs = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bi, br = imagined[idx], real[idx]
        if np.std(bi) < 1e-12 or np.std(br) < 1e-12:
            boot_rs[b] = np.nan
            continue
        boot_rs[b], _ = stats.pearsonr(bi, br)
    valid = boot_rs[~np.isnan(boot_rs)]
    alpha = (1.0 - ci_level) / 2.0
    if len(valid) == 0:
        ci_low = ci_high = float("nan")
        boot_mean = float("nan")
    else:
        ci_low = float(np.quantile(valid, alpha))
        ci_high = float(np.quantile(valid, 1.0 - alpha))
        boot_mean = float(np.mean(valid))

    return CorrelationResult(
        pearson_r=float(pearson_r),
        pearson_p_value=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_p_value=float(spearman_p),
        n_pairs=n,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        bootstrap_mean_r=boot_mean,
        strength=classify_correlation_strength(float(pearson_r)),
        imagined=imagined.tolist(),
        real=real.tolist(),
        labels=list(labels),
    )


def save_correlation(result: CorrelationResult, path: pathlib.Path) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(result.to_dict(), indent=2))


def save_scatter_plot(
    result: CorrelationResult,
    path: pathlib.Path,
    title: str = "Imagined vs Real Improvement",
) -> None:
    """Save a scatter plot of imagined vs real. Requires matplotlib."""
    # Late import so the module loads in environments without matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(result.imagined, result.real, s=60, alpha=0.7, edgecolor="black")

    # Label each point
    for x, y, lbl in zip(result.imagined, result.real, result.labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

    # Fit line if we have enough non-degenerate data
    if result.n_pairs >= 2 and np.std(result.imagined) > 1e-12:
        coeffs = np.polyfit(result.imagined, result.real, 1)
        xs = np.linspace(min(result.imagined), max(result.imagined), 100)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", alpha=0.5, label="fit")

    title_line = (
        f"{title}\n"
        f"Pearson r = {result.pearson_r:.3f} "
        f"[{result.ci_low:.3f}, {result.ci_high:.3f}]  "
        f"n={result.n_pairs}  strength={result.strength}"
    )
    ax.set_title(title_line, fontsize=10)
    ax.set_xlabel("Imagined improvement (world-model score)")
    ax.set_ylabel("Real success rate (LIBERO)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pair_by_candidate_and_task(
    imagined_per_candidate_task: dict[str, dict[int, float]],
    real_per_candidate_task: dict[str, dict[int, float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align imagined and real scores by (candidate, task) keys.

    Args:
        imagined_per_candidate_task: {candidate_name: {task_id: imagined_score}}
        real_per_candidate_task: {candidate_name: {task_id: real_success_rate}}

    Returns:
        (imagined, real, labels) as numpy arrays + list of "candidate@task" labels.
        Only pairs present in BOTH dicts are included.
    """
    imagined_list: list[float] = []
    real_list: list[float] = []
    labels: list[str] = []

    for cand in sorted(set(imagined_per_candidate_task) & set(real_per_candidate_task)):
        imag_tasks = imagined_per_candidate_task[cand]
        real_tasks = real_per_candidate_task[cand]
        for task_id in sorted(set(imag_tasks) & set(real_tasks)):
            imagined_list.append(float(imag_tasks[task_id]))
            real_list.append(float(real_tasks[task_id]))
            labels.append(f"{cand}@t{task_id}")

    return (
        np.asarray(imagined_list, dtype=np.float64),
        np.asarray(real_list, dtype=np.float64),
        labels,
    )
