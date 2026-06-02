"""Unsupervised re-stratification of failure traces by ε signature.

Hypothesis: BDDL outcome labels (IN_FALSE_CLOSE_FALSE etc) describe the
*world state at the end of rollout*. They mix multiple failure mechanisms.
Prediction-error signatures describe *what the policy did during the rollout*.
A clustering on ε signatures should yield a cleaner mechanism-level stratum,
specifically by splitting IN_FALSE_CLOSE_FALSE into execution-stuck vs
generation-level sub-clusters.

Method:
  1. Load per-trace ε feature vectors from `prediction_error_analysis/summary.json`.
  2. Restrict to the 26 failure traces (drop PHYS_OK).
  3. Standardize features and project to 2-D for visualization (PCA).
  4. Try K = 2, 3, 4 with KMeans + AgglomerativeClustering + GMM.
     Compare via silhouette score. Pick best.
  5. Compute confusion matrix (BDDL label × cluster).
  6. For each cluster, report mean feature values + 2 representative traces.

Usage:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/cluster_failures_by_epsilon.py
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "clustering"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    "corr_x", "corr_y", "corr_z",
    "mean_cos_align",
    "median_exec_ratio",
    "frac_low_exec_ratio",
    "max_consecutive_low_exec",
    "persistent_dir_after_high_eps",
]


def main() -> None:
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    rows = summary["rows"]

    failure_rows = [r for r in rows if r["label"] != "PHYS_OK"]
    print(f"loaded {len(failure_rows)} failure traces")
    print("BDDL label distribution:", Counter(r["label"] for r in failure_rows))

    X = np.array([[r[f] for f in FEATURE_NAMES] for r in failure_rows], dtype=float)
    # Replace NaNs (rare — degenerate corr) with feature median
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.any(np.isnan(col)):
            X[np.isnan(col), j] = np.nanmedian(col)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    bddl_labels = np.array([r["label"] for r in failure_rows])
    trace_names = [r["path"] for r in failure_rows]

    # Pick K by silhouette across multiple algorithms
    results = {}
    print("\nSilhouette scores (higher = better-separated clusters):")
    print(f"{'method':25s} " + " ".join(f"{'K=' + str(k):>10s}" for k in [2, 3, 4]))
    for method_name, make in [
        ("KMeans", lambda k: KMeans(n_clusters=k, n_init=20, random_state=0)),
        ("Ward (agglomerative)", lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ("GMM", lambda k: GaussianMixture(n_components=k, random_state=0, n_init=10)),
    ]:
        row_str = f"{method_name:25s} "
        for k in [2, 3, 4]:
            model = make(k)
            if isinstance(model, GaussianMixture):
                labels = model.fit_predict(Xs)
            else:
                labels = model.fit_predict(Xs)
            if len(set(labels)) < 2:
                row_str += f"{'n/a':>10s} "
                continue
            score = silhouette_score(Xs, labels)
            row_str += f"{score:>10.3f} "
            results[(method_name, k)] = (score, labels)
        print(row_str)

    # Pick best: highest silhouette across all (method, k) combos
    best_key = max(results, key=lambda k: results[k][0])
    best_score, best_labels = results[best_key]
    print(f"\nBest: {best_key[0]}, K={best_key[1]}, silhouette={best_score:.3f}")

    # Confusion matrix: BDDL label × cluster
    print("\nBDDL label × cluster assignment:")
    bddl_set = sorted(set(bddl_labels))
    cluster_set = sorted(set(best_labels))
    sep = "BDDL\\cluster"
    header = f"{sep:30s}" + "".join(f"{'C' + str(c):>6s}" for c in cluster_set)
    print(header)
    for b in bddl_set:
        row = f"{b:30s}"
        for c in cluster_set:
            n = int(np.sum((bddl_labels == b) & (best_labels == c)))
            row += f"{n:>6d}"
        print(row)

    # Per-cluster feature means (in original units, not standardized)
    print("\nPer-cluster feature means:")
    print(f"{'feature':35s}" + "".join(f"{'C' + str(c):>12s}" for c in cluster_set))
    for j, f in enumerate(FEATURE_NAMES):
        row = f"{f:35s}"
        for c in cluster_set:
            mask = best_labels == c
            row += f"{X[mask, j].mean():>12.3f}"
        print(row)

    # Representative trace per cluster (closest to cluster centroid in standardized space)
    print("\nRepresentative traces per cluster (closest to centroid):")
    for c in cluster_set:
        mask = best_labels == c
        if mask.sum() == 0:
            continue
        centroid = Xs[mask].mean(axis=0)
        dists = np.linalg.norm(Xs[mask] - centroid, axis=1)
        idx_in_cluster = np.argsort(dists)[:2]
        cluster_indices = np.where(mask)[0]
        print(f"  C{c} (N={mask.sum()}):")
        for ii in idx_in_cluster:
            global_idx = cluster_indices[ii]
            print(f"    {trace_names[global_idx]:80s} BDDL={bddl_labels[global_idx]}")

    # 2-D projection for visualization
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xs)
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # left: by BDDL label
    bddl_colors = {"IN_TRUE_CLOSE_FALSE": "tab:red",
                   "IN_FALSE_CLOSE_FALSE": "tab:blue",
                   "IN_FALSE_CLOSE_TRUE": "tab:green"}
    for b in bddl_set:
        mask = bddl_labels == b
        axes[0].scatter(X2[mask, 0], X2[mask, 1], c=bddl_colors.get(b, "gray"),
                        label=f"{b} (N={mask.sum()})", s=80, edgecolor="black", alpha=0.8)
    axes[0].set_title(f"PC-projected failures, colored by BDDL outcome label")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # right: by ε-cluster
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(2, len(cluster_set))))
    for ci, c in enumerate(cluster_set):
        mask = best_labels == c
        axes[1].scatter(X2[mask, 0], X2[mask, 1], c=[cluster_colors[ci]],
                        label=f"C{c} (N={mask.sum()})", s=80, edgecolor="black", alpha=0.8)
    axes[1].set_title(f"PC-projected failures, colored by ε-cluster ({best_key[0]}, K={best_key[1]})")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_bddl_vs_eps.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'pca_bddl_vs_eps.png'}")

    # Quantify how well each labeling separates the data, via silhouette in standardized space
    sil_bddl = silhouette_score(Xs, bddl_labels) if len(bddl_set) > 1 else float("nan")
    sil_cluster = best_score
    print(f"\nSilhouette score (higher = more internally-consistent groups):")
    print(f"  BDDL labels:  {sil_bddl:.3f}")
    print(f"  ε-clusters:   {sil_cluster:.3f}")
    print(f"  Δ:            {sil_cluster - sil_bddl:+.3f}  ({'ε wins' if sil_cluster > sil_bddl else 'BDDL wins'})")

    # Save assignments per trace
    assignments = [
        {"trace": trace_names[i], "bddl_label": bddl_labels[i],
         "eps_cluster": int(best_labels[i]), "features": {f: float(X[i, j]) for j, f in enumerate(FEATURE_NAMES)}}
        for i in range(len(failure_rows))
    ]
    (OUT_DIR / "assignments.json").write_text(json.dumps({
        "method": best_key[0], "K": best_key[1],
        "silhouette_bddl": float(sil_bddl) if not np.isnan(sil_bddl) else None,
        "silhouette_eps_cluster": float(sil_cluster),
        "assignments": assignments,
    }, indent=2))
    print(f"saved {OUT_DIR / 'assignments.json'}")


if __name__ == "__main__":
    main()
