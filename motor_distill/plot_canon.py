import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = [0, 6, 12, 16]
L1 = [0.733, 0.633, 0.300, 0.333]
pi05 = [0.967, 0.700, 0.833, 0.333]
canon = [1.000, 0.800, 0.667, 0.667]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
ax.plot(d, canon, "o-", color="#27ae60", lw=3, ms=10, label="OURS + SE(2) canonicalization (π0-free)")
ax.plot(d, pi05, "s--", color="#2471a3", lw=2, ms=7, label="π0.5 (goal-correcting ref)")
ax.plot(d, L1, "^--", color="#c0392b", lw=2, ms=7, label="OURS, L1 (goal-following)")
for x, y in zip(d, canon): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#27ae60", weight="bold")
ax.set_xlabel("object displacement from training position (cm)", fontsize=11)
ax.set_ylabel("grasp success (official metric)", fontsize=11)
ax.set_title("Position-shift extrapolation: canonicalization flattens the curve\n(beats π0.5 at 0/6/16cm; holds 0.67 where both collapse to 0.33)", fontsize=10.5)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="lower left")

labels = ["π0.5\nbaseline", "VLS\n(SOTA)", "ours\nL1", "ours\nCANON"]
vals = [0.17, 0.3681, 0.50, 0.80]
cols = ["#7f8c8d", "#e67e22", "#c0392b", "#27ae60"]
bars = ax2.bar(labels, vals, color=cols)
ax2.axhline(0.3681, ls=":", color="#e67e22", lw=1.2)
for b, v in zip(bars, vals): ax2.annotate(f"{v:.2f}", (b.get_x()+b.get_width()/2, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10, weight="bold")
ax2.set_ylabel("success", fontsize=11); ax2.set_ylim(0, 0.95)
ax2.set_title("LIBERO-PRO SWAP axis\n(object relocation, privileged goal)", fontsize=10.5)
ax2.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig("/Users/arifahmed/projects/openpi/canon_breakthrough.png", dpi=140)
print("saved canon_breakthrough.png")
