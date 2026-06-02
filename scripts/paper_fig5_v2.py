"""Updated Fig 5: net effect vs base failure rate with all 3 empirical points + pooled architecture math."""
import numpy as np
import matplotlib.pyplot as plt

# Pooled architecture parameters
R_disp = 0.80; D_disp = 0.064
R_nodisp = 0.70; D_nodisp = 0.10

p_fail = np.linspace(0, 0.30, 200)
net_disp = R_disp * p_fail - D_disp * (1 - p_fail)
net_nodisp = R_nodisp * p_fail - D_nodisp * (1 - p_fail)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(p_fail * 100, net_disp * 100, color="tab:green", linewidth=2.5,
        label=f"With dispatch (pooled): R={R_disp:.0%}, D={D_disp:.1%}")
ax.plot(p_fail * 100, net_nodisp * 100, color="tab:red", linewidth=2.5,
        label=f"No dispatch: R={R_nodisp:.0%}, D={D_nodisp:.0%}")
ax.axhline(0, color="black", alpha=0.3, linestyle="--")

# Empirical points: dispatched
points_disp = [
    ("5cm seed=1234", 11.1, +3.3),
    ("10cm seed=1234", 5.6, -2.2),
    ("5cm seed=42", 5.6, -1.1),
]
for label, x, y in points_disp:
    ax.scatter(x, y, marker="*", s=220, color="tab:green", edgecolor="black", zorder=5)
    ax.annotate(label, (x, y), fontsize=8, xytext=(5, 7), textcoords="offset points")

# Empirical points: no-dispatch
points_nodisp = [
    ("5cm no-disp", 11.1, -1.1),
    ("10cm no-disp", 5.6, -6.7),
]
for label, x, y in points_nodisp:
    ax.scatter(x, y, marker="o", s=130, color="tab:red", edgecolor="black", zorder=5)
    ax.annotate(label, (x, y), fontsize=8, xytext=(5, -12), textcoords="offset points")

# Break-even lines
be_disp = D_disp / (R_disp + D_disp)
be_nodisp = D_nodisp / (R_nodisp + D_nodisp)
ax.axvline(be_disp * 100, color="tab:green", linestyle=":", alpha=0.6,
           label=f"Dispatch break-even: {be_disp*100:.1f}%")
ax.axvline(be_nodisp * 100, color="tab:red", linestyle=":", alpha=0.6,
           label=f"No-dispatch break-even: {be_nodisp*100:.1f}%")

ax.set_xlabel("Base failure rate P(fail), %", fontsize=12)
ax.set_ylabel("Net effect on success rate, pp", fontsize=12)
ax.set_title("Architecture math validated across 3 seed/perturbation configurations\n"
             "Mechanism dispatch lowers break-even from 12.5% to 7.4%", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 25)
ax.set_ylim(-10, +12)
fig.tight_layout()
fig.savefig("/Users/arifahmed/projects/openpi/paper_figures/fig5_architecture_math_pooled.png", dpi=150, bbox_inches="tight")
print("saved fig5_architecture_math_pooled.png")
