import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = [0, 3, 6, 9, 12, 16]
old = [0.733, 0.633, 0.633, 0.433, 0.300, 0.333]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(d, old, "o-", color="#c0392b", lw=2.4, ms=8, label="relative-goal wrist motor (standard-only)")
ax.axhline(0.733, ls="--", color="#27ae60", lw=1.6, label="if perfectly GENERAL (flat @ d=0 level)")
ax.axhline(0.167, ls=":", color="#7f8c8d", lw=1.6, label="blind / chance (16.7%)")
ax.fill_between([0, 16], 0.167, 0.733, color="#27ae60", alpha=0.05)
for x, y in zip(d, old):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#c0392b")
ax.set_xlabel("object displacement from training position  (cm)", fontsize=11)
ax.set_ylabel("grasp success  (official metric)", fontsize=11)
ax.set_title("Position-shift generalization curve  —  EXTRAPOLATION test\n"
             "correct relative goal at every position; distractors hidden", fontsize=11)
ax.set_ylim(0, 0.85); ax.set_xlim(-0.5, 16.5)
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper right")
txt = ("DECAY (Δ = −0.40) → position leaks in: NOT fully general.\n"
       "But graceful (16cm still 2× chance) → not a hard lookup table.\n"
       "Adding position-diverse data did NOT help (swap 57%→43%) → leak is ARCHITECTURAL.")
ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.2, va="bottom",
        bbox=dict(boxstyle="round", fc="#fdf2e9", ec="#e67e22", alpha=0.9))
fig.tight_layout()
fig.savefig("/Users/arifahmed/projects/openpi/posshift_curve.png", dpi=140)
print("saved posshift_curve.png")
