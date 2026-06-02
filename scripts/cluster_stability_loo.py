"""Leave-one-out and feature-subset cluster stability tests on the 26 failure traces.

Asks: how robust is the K=2 cluster structure on this dataset?

Two stability tests:
  (a) Leave-one-trace-out (LOO): for each trace, hold it out, re-cluster the
      remaining 25 with KMeans K=2. Compare assignments on the 25 to the
      baseline 26-trace assignments via Adjusted Rand Index (ARI).
      Outputs: distribution of ARIs, and a list of traces whose held-out
      assignment changed under LOO.
  (b) Feature-subset stability: drop each feature one at a time, re-cluster,
      check ARI vs baseline. Tells us which features are load-bearing.
  (c) Bootstrap stability: resample-with-replacement from the 26, re-cluster.

Run: /Users/arifahmed/projects/openpi/.venv/bin/python scripts/cluster_stability_loo.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "clustering"
FEATURES = [
    "corr_x", "corr_y", "corr_z",
    "mean_cos_align", "median_exec_ratio",
    "frac_low_exec_ratio", "max_consecutive_low_exec",
    "persistent_dir_after_high_eps",
]


def cluster_kmeans(X: np.ndarray, k: int = 2, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)


def align_labels(reference: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Permute `new` labels to maximize agreement with `reference`. K=2 only."""
    if len(set(new)) < 2:
        return new
    # try identity vs flip
    flipped = 1 - new
    if (new == reference).sum() >= (flipped == reference).sum():
        return new
    return flipped


def main() -> None:
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    rows = [r for r in summary["rows"] if r["label"] != "PHYS_OK"]
    print(f"loaded {len(rows)} failure traces")

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    for j in range(X.shape[1]):
        if np.any(np.isnan(X[:, j])):
            X[np.isnan(X[:, j]), j] = np.nanmedian(X[:, j])
    Xs = StandardScaler().fit_transform(X)

    baseline = cluster_kmeans(Xs)
    print(f"baseline cluster sizes: {np.bincount(baseline)}")

    # --- (a) Leave-one-out ---
    print("\n=== Leave-one-out trace stability ===")
    aris = []
    changed_assignments = 0
    detailed = []
    for i in range(len(rows)):
        keep = np.delete(np.arange(len(rows)), i)
        Xs_loo = Xs[keep]
        labels_loo = cluster_kmeans(Xs_loo, seed=0)
        labels_loo_aligned = align_labels(baseline[keep], labels_loo)
        ari = adjusted_rand_score(baseline[keep], labels_loo_aligned)
        aris.append(ari)
        baseline_label = int(baseline[i])
        # what does the LOO-fit clusterer assign to the held-out trace?
        loo_model = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xs_loo)
        held_out_label = int(loo_model.predict(Xs[i:i+1])[0])
        # align held-out label using the same flip as for the LOO labels
        # (recompute flip)
        flipped_loo = 1 - labels_loo
        flip = (flipped_loo == baseline[keep]).sum() > (labels_loo == baseline[keep]).sum()
        if flip:
            held_out_label = 1 - held_out_label
        if held_out_label != baseline_label:
            changed_assignments += 1
            detailed.append({"trace": rows[i]["path"], "baseline_cluster": baseline_label,
                             "loo_cluster": held_out_label, "bddl_label": rows[i]["label"]})

    aris = np.array(aris)
    print(f"ARI distribution: mean={aris.mean():.3f}  median={np.median(aris):.3f}  min={aris.min():.3f}  max={aris.max():.3f}")
    print(f"# held-out traces with changed cluster assignment: {changed_assignments}/{len(rows)}")
    if detailed:
        print("Changed traces:")
        for d in detailed:
            print(f"  {d['trace']:80s} BDDL={d['bddl_label']:25s} baseline=C{d['baseline_cluster']} → LOO=C{d['loo_cluster']}")

    # --- (b) Feature-subset stability ---
    print("\n=== Drop-one-feature stability ===")
    for j, fname in enumerate(FEATURES):
        keep_features = [k for k in range(len(FEATURES)) if k != j]
        Xs_drop = Xs[:, keep_features]
        labels_drop = cluster_kmeans(Xs_drop)
        labels_drop_aligned = align_labels(baseline, labels_drop)
        ari = adjusted_rand_score(baseline, labels_drop_aligned)
        n_changed = int((labels_drop_aligned != baseline).sum())
        print(f"  drop {fname:35s}  ARI={ari:.3f}  #changed_assignments={n_changed}/{len(rows)}")

    # --- (c) Bootstrap stability ---
    print("\n=== Bootstrap resample stability (N=200) ===")
    rng = np.random.default_rng(0)
    boot_aris = []
    for _ in range(200):
        idx = rng.integers(0, len(rows), size=len(rows))
        labels_boot = cluster_kmeans(Xs[idx])
        if len(set(labels_boot)) < 2:
            continue
        # We can only compare on the indices that appear; multi-mapping aware ARI
        # is overkill — simpler: compare baseline[idx] vs labels_boot.
        labels_boot_aligned = align_labels(baseline[idx], labels_boot)
        boot_aris.append(adjusted_rand_score(baseline[idx], labels_boot_aligned))
    boot_aris = np.array(boot_aris)
    print(f"Bootstrap ARI: mean={boot_aris.mean():.3f}  median={np.median(boot_aris):.3f}  "
          f"2.5%={np.percentile(boot_aris, 2.5):.3f}  97.5%={np.percentile(boot_aris, 97.5):.3f}")

    # Save
    (OUT_DIR / "stability.json").write_text(json.dumps({
        "loo_ari": {"mean": float(aris.mean()), "median": float(np.median(aris)),
                    "min": float(aris.min()), "max": float(aris.max()),
                    "n_changed": int(changed_assignments),
                    "changed_traces": detailed},
        "bootstrap_ari": {"mean": float(boot_aris.mean()),
                          "ci": [float(np.percentile(boot_aris, 2.5)),
                                 float(np.percentile(boot_aris, 97.5))]},
    }, indent=2))
    print(f"\nsaved {OUT_DIR / 'stability.json'}")


if __name__ == "__main__":
    main()
