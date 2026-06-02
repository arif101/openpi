"""Phase 1.5 sanity checks before committing to Phase 2.

Three blocks:

(A) **C0 predictability check.** The Phase-1 finding that the self-model has
    R²=0.04 on C0 traces could mean either "different mechanism, different
    signal needed" (Head B story) OR "C0 is inherently unpredictable from any
    observable, Head B will also fail" (escalation story).
    To disambiguate, train a binary classifier (PHYS_OK vs C0) on features
    OTHER than ε — proxies for object progress, motion, contact, trace shape.
    If AUROC > ~0.7, C0 has learnable structure → Head B is plausible.
    If AUROC ≈ 0.5, C0 is structureless → architecture should triage, not split.

    Mirror check on PHYS_OK vs C1: confirms the comparison framework works.

(B) **ARI degeneracy check.** ARI=1.000 LOO is suspicious. Three checks:
    (1) silhouette curve K=2..8 — does K=2 truly peak, or is the curve flat/
        monotone (degenerate)?
    (2) feature dominance: does max_consecutive_low_exec alone determine
        cluster, or is it joint?
    (3) per-cluster within-cluster variance vs between-cluster gap — is the
        boundary wide or driven by a couple of outliers?

(C) **Sentinel STAC reproduction (proxy).** We don't have policy action
    distributions, but we have executed action chunks. Cosine similarity
    between consecutive chunk directions is a reasonable proxy for STAC:
    high cos = consistent policy, low cos = erratic. Per Sentinel, this
    should fire on erratic failures and miss the temporally-consistent
    IN_TRUE_CLOSE_FALSE drawer-close mode. Test this directly.

Run:
    /Users/arifahmed/projects/openpi/.venv/bin/python scripts/phase15_sanity_checks.py
"""
from __future__ import annotations

import json
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

TRACES_DIR = pathlib.Path("data/contact_mpc/recovery_source_traces")
ANALYSIS_DIR = pathlib.Path("data/contact_mpc/prediction_error_analysis")
OUT_DIR = ANALYSIS_DIR / "phase15_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_HORIZON = 10
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# Block A: C0 predictability from non-ε features
# --------------------------------------------------------------------------

def extract_non_eps_features(path: pathlib.Path) -> dict | None:
    d = np.load(path, allow_pickle=True)
    action = d["action"]                 # (T, 7)
    ee_pos = d["ee_pos"]                 # (T, 3)
    object_pos = d["object_pos"]         # (T, n_obj, 3)
    object_names = d["object_names"]     # (n_obj,)
    gripper = d["gripper_qpos"]          # (T, 2)
    contacts = d["contacts"]             # (T,) object array
    T = action.shape[0]
    if T < 5:
        return None

    realized = np.diff(ee_pos, axis=0)            # (T-1, 3)
    realized_mag = np.linalg.norm(realized, axis=1)

    # Object-distance proxies — pick the manipulated object (bowl, in task 3).
    # We don't know the target site, but we can use the EE-to-object distance
    # as a "approached the object" proxy and object motion magnitude.
    # Pick the most-moved object as the manipulated one (heuristic for task-3).
    obj_motion = np.linalg.norm(np.diff(object_pos, axis=0), axis=2)  # (T-1, n_obj)
    manip_obj_idx = int(obj_motion.sum(axis=0).argmax())
    obj_xyz = object_pos[:, manip_obj_idx, :]              # (T, 3)
    ee_to_obj = np.linalg.norm(ee_pos - obj_xyz, axis=1)   # (T,)
    obj_displacement_total = float(np.linalg.norm(obj_xyz[-1] - obj_xyz[0]))
    obj_displacement_max = float(np.linalg.norm(obj_xyz - obj_xyz[0], axis=1).max())

    # Contact-event count: how many timesteps had non-empty contact list
    contact_count = 0
    for c in contacts:
        try:
            if c is not None and len(c) > 0:
                contact_count += 1
        except TypeError:
            pass

    feats = {
        "trace_length": float(T),
        "realized_motion_total": float(realized_mag.sum()),
        "realized_motion_mean": float(realized_mag.mean()),
        "realized_motion_std": float(realized_mag.std()),
        "ee_to_obj_mean": float(ee_to_obj.mean()),
        "ee_to_obj_min": float(ee_to_obj.min()),
        "ee_to_obj_final": float(ee_to_obj[-1]),
        "obj_displacement_total": obj_displacement_total,
        "obj_displacement_max": obj_displacement_max,
        "contact_step_count": float(contact_count),
        "contact_step_frac": float(contact_count / T),
        "gripper_action_var": float(np.var(action[:, 6])),
        "gripper_state_var": float(np.var(gripper[:, 0])),
        "action_xyz_norm_mean": float(np.linalg.norm(action[:, :3], axis=1).mean()),
        "action_xyz_norm_std": float(np.linalg.norm(action[:, :3], axis=1).std()),
    }
    return feats


def block_a_c0_predictability() -> dict:
    print("\n" + "=" * 70)
    print("BLOCK A: C0 predictability from non-ε features")
    print("=" * 70)
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    feats_per_trace = {}
    for p in sorted(TRACES_DIR.glob("*.npz")):
        f = extract_non_eps_features(p)
        if f is None:
            continue
        feats_per_trace[p.name] = f

    feature_names = list(next(iter(feats_per_trace.values())).keys())
    print(f"computed {len(feature_names)} non-ε features for {len(feats_per_trace)} traces")
    print(f"features: {feature_names}\n")

    # Build labels
    rows = []
    for name, feats in feats_per_trace.items():
        if name.startswith("PHYS_OK_"):
            label = "PHYS_OK"
        elif name in cluster_of:
            label = f"C{cluster_of[name]}"
        else:
            label = "UNK"
        rows.append((name, label, [feats[k] for k in feature_names]))

    # Contrast 1: PHYS_OK vs C0
    contrasts = [
        ("PHYS_OK", "C0", "Head-B target (C0 = generation failure)"),
        ("PHYS_OK", "C1", "Head-A target (C1 = execution stuck)"),
        ("C0", "C1", "Mechanism contrast (C0 vs C1)"),
    ]

    results = {}
    for neg, pos, descr in contrasts:
        neg_rows = [r for r in rows if r[1] == neg]
        pos_rows = [r for r in rows if r[1] == pos]
        if len(neg_rows) < 2 or len(pos_rows) < 2:
            print(f"  {neg} vs {pos}: insufficient samples")
            continue

        # Per-feature AUROC (no model, just univariate discriminability)
        y = np.array([0] * len(neg_rows) + [1] * len(pos_rows))
        X = np.array([r[2] for r in neg_rows + pos_rows])

        print(f"\n--- {neg} (N={len(neg_rows)}) vs {pos} (N={len(pos_rows)}): {descr} ---")
        per_feature = {}
        for j, fname in enumerate(feature_names):
            scores = X[:, j]
            # Use the sign that gives AUROC >= 0.5
            auc = roc_auc_score(y, scores)
            if auc < 0.5:
                auc = 1.0 - auc
                sign = "-"
            else:
                sign = "+"
            per_feature[fname] = (auc, sign)

        # Sort by AUROC
        for fname, (auc, sign) in sorted(per_feature.items(), key=lambda x: -x[1][0]):
            print(f"    {fname:30s}  AUROC={auc:.3f}  sign={sign}")

        # Multi-feature classifier (LOO logistic regression on all features)
        Xs = StandardScaler().fit_transform(X)
        loo = LeaveOneOut()
        probs = np.zeros(len(y))
        for tr, te in loo.split(Xs):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xs[tr], y[tr])
            probs[te] = clf.predict_proba(Xs[te])[:, 1]
        multi_auc = roc_auc_score(y, probs)
        print(f"    >> LOO multi-feature AUROC: {multi_auc:.3f}")
        results[f"{neg}_vs_{pos}"] = {
            "per_feature": {k: {"auroc": v[0], "sign": v[1]} for k, v in per_feature.items()},
            "multi_feature_auroc_loo": float(multi_auc),
            "n_neg": len(neg_rows), "n_pos": len(pos_rows),
        }
    return results


# --------------------------------------------------------------------------
# Block B: ARI degeneracy / K-sweep / cluster gap
# --------------------------------------------------------------------------

EPS_FEATURES = [
    "corr_x", "corr_y", "corr_z", "mean_cos_align", "median_exec_ratio",
    "frac_low_exec_ratio", "max_consecutive_low_exec",
    "persistent_dir_after_high_eps",
]


def block_b_ari_degeneracy() -> dict:
    print("\n" + "=" * 70)
    print("BLOCK B: ARI degeneracy / cluster structure sanity")
    print("=" * 70)
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    fail_rows = [r for r in summary["rows"] if r["label"] != "PHYS_OK"]
    X = np.array([[r[f] for f in EPS_FEATURES] for r in fail_rows], dtype=float)
    for j in range(X.shape[1]):
        if np.any(np.isnan(X[:, j])):
            X[np.isnan(X[:, j]), j] = np.nanmedian(X[:, j])
    Xs = StandardScaler().fit_transform(X)

    # (B1) K-sweep silhouette
    print("\n(B1) Silhouette curve K=2..8 (should peak at K=2 if our claim is real):")
    silhouettes = {}
    for k in range(2, 9):
        if len(Xs) <= k:
            continue
        labels = KMeans(n_clusters=k, n_init=30, random_state=0).fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(Xs, labels)
        silhouettes[k] = float(s)
        print(f"  K={k}: silhouette={s:.3f}  cluster sizes={np.bincount(labels)}")

    # (B2) Single-feature clustering: does ONE feature determine cluster?
    print("\n(B2) Single-feature cluster recovery (do we just rediscover one feature?):")
    baseline_labels = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(Xs)
    from sklearn.metrics import adjusted_rand_score
    single_feature_ari = {}
    for j, fname in enumerate(EPS_FEATURES):
        # Cluster on just this feature
        Xs_single = Xs[:, j:j+1]
        labels_single = KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(Xs_single)
        # Align labels
        if (labels_single == baseline_labels).sum() < (1 - labels_single == baseline_labels).sum():
            labels_single = 1 - labels_single
        ari = adjusted_rand_score(baseline_labels, labels_single)
        single_feature_ari[fname] = float(ari)
        print(f"  {fname:35s}  ARI vs 8-feature baseline: {ari:.3f}")

    # (B3) Within-cluster vs between-cluster gap
    print("\n(B3) Within-cluster vs between-cluster gap (raw feature scales):")
    print(f"{'feature':35s} {'C0 mean':>12s} {'C0 std':>10s} {'C1 mean':>12s} {'C1 std':>10s} {'gap/std':>10s}")
    gap_summary = {}
    for j, fname in enumerate(EPS_FEATURES):
        v = X[:, j]
        c0_v = v[baseline_labels == 0]
        c1_v = v[baseline_labels == 1]
        pooled_std = np.sqrt((c0_v.std() ** 2 + c1_v.std() ** 2) / 2 + 1e-9)
        gap_units = abs(c0_v.mean() - c1_v.mean()) / pooled_std
        gap_summary[fname] = float(gap_units)
        print(f"  {fname:35s}  {c0_v.mean():>12.3f}  {c0_v.std():>10.3f}  "
              f"{c1_v.mean():>12.3f}  {c1_v.std():>10.3f}  {gap_units:>10.2f}")

    # Plot the silhouette curve and one feature distribution split by cluster
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ks = sorted(silhouettes.keys())
    axes[0].plot(ks, [silhouettes[k] for k in ks], "o-")
    axes[0].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="0.5 reference")
    axes[0].set_xlabel("K (n_clusters)")
    axes[0].set_ylabel("silhouette score")
    axes[0].set_title("K-sweep silhouette curve\n(should peak at K=2)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Most discriminative feature, distribution by cluster
    best_feat = max(gap_summary, key=gap_summary.get)
    j = EPS_FEATURES.index(best_feat)
    axes[1].hist(X[baseline_labels == 0, j], bins=8, alpha=0.6, label=f"C0 (N={(baseline_labels==0).sum()})", color="tab:blue")
    axes[1].hist(X[baseline_labels == 1, j], bins=8, alpha=0.6, label=f"C1 (N={(baseline_labels==1).sum()})", color="tab:orange")
    axes[1].set_xlabel(best_feat)
    axes[1].set_ylabel("# traces")
    axes[1].set_title(f"Most discriminative feature: {best_feat}\n"
                      f"(gap = {gap_summary[best_feat]:.2f} pooled std)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ari_sanity.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved {OUT_DIR / 'ari_sanity.png'}")

    return {
        "k_sweep_silhouette": silhouettes,
        "single_feature_ari_vs_baseline": single_feature_ari,
        "gap_in_pooled_std": gap_summary,
        "most_discriminative_feature": best_feat,
    }


# --------------------------------------------------------------------------
# Block C: Sentinel STAC reproduction (proxy)
# --------------------------------------------------------------------------

def block_c_sentinel_stac() -> dict:
    print("\n" + "=" * 70)
    print("BLOCK C: Sentinel STAC proxy reproduction")
    print("=" * 70)
    print("STAC ≈ statistical temporal action consistency. We don't have policy")
    print("action distributions (Pi0.5 emits chunks), so use chunk-to-chunk")
    print("cosine similarity as a proxy. HIGH cos = consistent policy (Sentinel")
    print("says 'fine'). LOW cos = erratic (Sentinel says 'failing').")
    print()
    print("Hypothesis: STAC misses IN_TRUE_CLOSE_FALSE because the policy is")
    print("TEMPORALLY CONSISTENT (keeps commanding +y across chunks).")

    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text())
    label_of = {r["path"]: r["label"] for r in summary["rows"]}
    assignments = json.loads((ANALYSIS_DIR / "clustering" / "assignments.json").read_text())
    cluster_of = {a["trace"]: a["eps_cluster"] for a in assignments["assignments"]}

    stac_per_trace = {}
    for p in sorted(TRACES_DIR.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        action = d["action"]
        T = action.shape[0]
        # For each adjacent chunk boundary, compute cosine similarity of mean
        # commanded direction. Lower mean cos = more erratic.
        chunk_starts = list(range(0, T, ACTION_HORIZON))
        chunk_dirs = []
        for c0 in chunk_starts:
            c1 = min(c0 + ACTION_HORIZON, T)
            if c1 - c0 < 2:
                continue
            v = action[c0:c1, :3].mean(axis=0)
            n = np.linalg.norm(v)
            if n > 1e-6:
                chunk_dirs.append(v / n)
        chunk_dirs = np.array(chunk_dirs)
        if len(chunk_dirs) < 2:
            continue
        cos_adj = np.einsum("ij,ij->i", chunk_dirs[:-1], chunk_dirs[1:])
        # Lower = more erratic. Sentinel-style score: 1 - mean(cos), high = erratic
        sentinel_score = float(1.0 - cos_adj.mean())
        stac_per_trace[p.name] = sentinel_score

    # Stratum aggregation
    by_stratum = defaultdict(list)
    for nm, s in stac_per_trace.items():
        if nm.startswith("PHYS_OK_"):
            by_stratum["PHYS_OK"].append(s)
        elif nm in cluster_of:
            by_stratum[f"C{cluster_of[nm]} ({label_of.get(nm,'?')})"].append(s)
        else:
            by_stratum[label_of.get(nm, "UNK")].append(s)

    by_simple = defaultdict(list)
    for nm, s in stac_per_trace.items():
        if nm.startswith("PHYS_OK_"):
            by_simple["PHYS_OK"].append(s)
        elif nm in cluster_of:
            by_simple[f"C{cluster_of[nm]}"].append(s)

    print("\nSentinel STAC proxy score (higher = more erratic) by stratum:")
    print(f"{'stratum':40s} {'N':>4s} {'mean':>10s} {'median':>10s} {'std':>10s}")
    for s in sorted(by_simple):
        v = np.array(by_simple[s])
        print(f"  {s:38s} {len(v):>4d}  {v.mean():>10.3f}  {np.median(v):>10.3f}  {v.std():>10.3f}")

    # AUROCs: Sentinel-style classifier vs each stratum
    print("\nSentinel STAC proxy AUROC (positive = right-side stratum):")
    cs = {s: np.array(by_simple[s]) for s in by_simple}
    contrasts = [
        ("PHYS_OK", "C0"),
        ("PHYS_OK", "C1"),
        ("C1", "C0"),  # is STAC able to separate the mechanisms?
    ]
    auroc_results = {}
    for neg, pos in contrasts:
        if neg not in cs or pos not in cs:
            continue
        y = np.array([0] * len(cs[neg]) + [1] * len(cs[pos]))
        scores = np.concatenate([cs[neg], cs[pos]])
        try:
            auc = roc_auc_score(y, scores)
            print(f"  STAC on {neg:8s} vs {pos:8s}:  AUROC = {auc:.3f}  (N={len(cs[neg])} vs {len(cs[pos])})")
            auroc_results[f"{neg}_vs_{pos}"] = float(auc)
        except Exception as e:
            print(f"  {e}")

    return {"by_stratum_sentinel_score": {k: list(v) for k, v in by_simple.items()},
            "aurocs": auroc_results}


# --------------------------------------------------------------------------

def main() -> None:
    results = {
        "block_a_c0_predictability": block_a_c0_predictability(),
        "block_b_ari_degeneracy": block_b_ari_degeneracy(),
        "block_c_sentinel_stac": block_c_sentinel_stac(),
    }
    (OUT_DIR / "phase15_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_DIR / 'phase15_results.json'}")


if __name__ == "__main__":
    main()
