"""Generate paper figures from accumulated Phase 2 results.

Figures:
  F1: Architecture diagram (text-based summary)
  F2: ε-cluster PCA (calibration corpus, 26 failures)
  F3: Gate firing distribution + lead-time histogram
  F4: Phase 2c vs Phase 2f transition matrices (heatmap)
  F5: Architecture math curve (net effect vs base failure rate)
"""
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LAPTOP = pathlib.Path("/Users/arifahmed/projects/openpi")
OUT = LAPTOP / "paper_figures"
OUT.mkdir(exist_ok=True)

EPS_FEATURES = ["corr_x","corr_y","corr_z","mean_cos_align","median_exec_ratio",
                "frac_low_exec_ratio","max_consecutive_low_exec","persistent_dir_after_high_eps"]


def fig2_eps_cluster_pca():
    """ε-cluster PCA from calibration corpus."""
    calib = json.loads((LAPTOP / "data/contact_mpc/prediction_error_analysis/summary.json").read_text())
    fails = [r for r in calib["rows"] if r["label"] != "PHYS_OK"]
    X = np.array([[r[f] for f in EPS_FEATURES] for r in fails])
    for j in range(X.shape[1]):
        if np.any(np.isnan(X[:, j])):
            X[np.isnan(X[:, j]), j] = np.nanmedian(X[:, j])
    Xs = StandardScaler().fit_transform(X)
    k = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xs)
    c1_idx = int(np.argmax([X[k.labels_==c, EPS_FEATURES.index("frac_low_exec_ratio")].mean() for c in range(2)]))
    pca = PCA(n_components=2).fit_transform(Xs)
    labels = ["C1" if c == c1_idx else "C0" for c in k.labels_]
    bddl_labels = [r["label"] for r in fails]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Left: colored by BDDL label
    bddl_colors = {"IN_TRUE_CLOSE_FALSE": "tab:red", "IN_FALSE_CLOSE_FALSE": "tab:blue", "IN_FALSE_CLOSE_TRUE": "tab:green"}
    for lab in sorted(set(bddl_labels)):
        mask = np.array(bddl_labels) == lab
        axes[0].scatter(pca[mask, 0], pca[mask, 1], c=bddl_colors.get(lab, "gray"),
                        label=f"{lab} (N={mask.sum()})", s=80, edgecolor="black", alpha=0.8)
    axes[0].set_title("PC-projected failures (calibration)\ncolored by BDDL outcome label", fontsize=11)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: colored by ε-cluster
    cluster_colors = {"C0": "tab:purple", "C1": "tab:orange"}
    for c in sorted(set(labels)):
        mask = np.array(labels) == c
        axes[1].scatter(pca[mask, 0], pca[mask, 1], c=cluster_colors[c],
                        label=f"{c} (N={mask.sum()})", s=80, edgecolor="black", alpha=0.8)
    axes[1].set_title("Same failures, colored by ε-signature cluster\n(silhouette 0.50 vs BDDL 0.10)", fontsize=11)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_eps_cluster_pca.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'fig2_eps_cluster_pca.png'}")


def fig4_transition_heatmaps():
    """Phase 2c vs Phase 2f trial-by-trial transitions, side by side."""
    # Read both paired analyses
    p2c = json.loads((LAPTOP / "data/contact_mpc/phase2b_gpu_results/phase2c_paired.json").read_text())
    p2f = json.loads((LAPTOP / "data/contact_mpc/phase2b_gpu_results/phase2f_dispatched_paired.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, label, data in [(axes[0], "Phase 2c: intervention on all gate fires", p2c),
                              (axes[1], "Phase 2f: dispatched (C1-only) intervention", p2f)]:
        T = data["transitions"]
        mat = np.array([
            [T.get("OK→OK", 0), T.get("OK→FAIL", 0)],
            [T.get("FAIL→OK", 0), T.get("FAIL→FAIL", 0)],
        ])
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Intervention OK", "Intervention FAIL"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Baseline OK", "Baseline FAIL"])
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=18, color=color, fontweight="bold")
        ax.set_title(f"{label}\nnet {data['net_effect_point']:+.3f}, recovery {data['recovery_rate_point']:.0%}, damage {data['damage_rate_point']:.0%}",
                     fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_transitions.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'fig4_transitions.png'}")


def fig5_architecture_math():
    """Net effect vs base failure rate, predicted vs observed."""
    rec_no_disp = 0.70
    dam_no_disp = 0.10
    rec_disp = 1.00
    dam_disp = 0.09

    p_fail = np.linspace(0, 0.5, 100)
    net_no_disp = rec_no_disp * p_fail - dam_no_disp * (1 - p_fail)
    net_disp = rec_disp * p_fail - dam_disp * (1 - p_fail)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_fail * 100, net_no_disp * 100, color="tab:red", linewidth=2,
            label=f"No dispatch: rec={rec_no_disp:.0%}, dam={dam_no_disp:.0%}")
    ax.plot(p_fail * 100, net_disp * 100, color="tab:green", linewidth=2,
            label=f"With dispatch: rec={rec_disp:.0%}, dam={dam_disp:.0%}")
    ax.axhline(0, color="black", alpha=0.3, linestyle="--")

    # Observed points
    obs_5cm_no_disp = (11.1, -1.1)
    obs_5cm_disp = (11.1, +3.3)
    obs_10cm_no_disp = (5.6, -6.7)
    ax.scatter(*obs_5cm_no_disp, marker="o", s=120, color="tab:red", edgecolor="black", zorder=5,
               label=f"5cm observed (no dispatch): {obs_5cm_no_disp[1]:+.1f}pp")
    ax.scatter(*obs_5cm_disp, marker="*", s=200, color="tab:green", edgecolor="black", zorder=5,
               label=f"5cm observed (dispatch): {obs_5cm_disp[1]:+.1f}pp")
    ax.scatter(*obs_10cm_no_disp, marker="s", s=120, color="tab:red", edgecolor="black", zorder=5,
               label=f"10cm observed (no dispatch): {obs_10cm_no_disp[1]:+.1f}pp")

    # Break-even lines
    be_no_disp = dam_no_disp / (rec_no_disp + dam_no_disp)
    be_disp = dam_disp / (rec_disp + dam_disp)
    ax.axvline(be_no_disp * 100, color="tab:red", linestyle=":", alpha=0.5, label=f"Break-even (no dispatch): {be_no_disp*100:.1f}%")
    ax.axvline(be_disp * 100, color="tab:green", linestyle=":", alpha=0.5, label=f"Break-even (dispatch): {be_disp*100:.1f}%")

    ax.set_xlabel("Base failure rate P(fail), %", fontsize=12)
    ax.set_ylabel("Net effect on success rate, pp", fontsize=12)
    ax.set_title("Architecture math: net effect = R·P(fail) − D·(1−P(fail))\nMechanism dispatch lowers damage → lowers break-even", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_architecture_math.png", dpi=150, bbox_inches="tight")
    print(f"saved {OUT / 'fig5_architecture_math.png'}")


if __name__ == "__main__":
    fig2_eps_cluster_pca()
    fig4_transition_heatmaps()
    fig5_architecture_math()
    print(f"\nAll figures in {OUT}")
