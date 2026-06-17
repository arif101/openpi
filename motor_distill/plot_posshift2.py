import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = [0, 6, 12, 16]
ours = [0.733, 0.633, 0.300, 0.333]
pi05 = [0.967, 0.700, 0.833, 0.333]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.plot(d, pi05, "s-", color="#2471a3", lw=2.6, ms=9, label="π0.5  (goal-CORRECTING: closed-loop wrist servo)")
ax.plot(d, ours, "o-", color="#c0392b", lw=2.6, ms=9, label="our motor  (goal-FOLLOWING: open-loop on goal)")
ax.axhline(0.167, ls=":", color="#7f8c8d", lw=1.4, label="blind / chance")
for x, y in zip(d, ours): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color="#c0392b")
for x, y in zip(d, pi05): ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color="#2471a3")
# highlight the addressable gap at 12cm
ax.annotate("", xy=(12, 0.833), xytext=(12, 0.300), arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=2))
ax.text(12.3, 0.56, "addressable\ngap +0.53", fontsize=8.5, color="#e67e22", va="center")
ax.axvspan(14.5, 16.5, color="#7f8c8d", alpha=0.10)
ax.text(15.5, 0.05, "reachability\nwall (both)", fontsize=7.6, color="#555", ha="center")
ax.set_xlabel("object displacement from training position  (cm)", fontsize=11)
ax.set_ylabel("grasp success  (official metric)", fontsize=11)
ax.set_title("Goal-CORRECTING vs goal-FOLLOWING: position-shift extrapolation\n"
             "correct relative goal + distractors hidden; identical harness", fontsize=11)
ax.set_ylim(0, 1.0); ax.set_xlim(-0.5, 16.8); ax.grid(alpha=0.25); ax.legend(fontsize=8.6, loc="upper center")
fig.tight_layout(); fig.savefig("/Users/arifahmed/projects/openpi/posshift_vs_pi05.png", dpi=140)
print("saved posshift_vs_pi05.png")
