import matplotlib.pyplot as plt
import numpy as np

# Data
N = np.arange(1, 7)
delta_S = np.array([0.50, 0.17, 0.08, 0.05, 0.03, 0.02])  # Search-time (ΔS)
delta_P = np.array([0.00, 0.03, 0.08, 0.15, 0.25, 0.40])  # Pursuit-time (ΔP)

fig, ax = plt.subplots(figsize=(6, 4))

# Draw grid behind
ax.set_axisbelow(True)

# ΔS: blue upward triangles
l1, = ax.plot(
    N, delta_S,
    marker='^', color='tab:blue',
    markerfacecolor='tab:blue', markeredgecolor='white',
    markersize=8, linewidth=1.5,
    label='Search-time (ΔS)'
)

# ΔP: orange downward triangles
l2, = ax.plot(
    N, delta_P,
    marker='v', color='tab:orange',
    markerfacecolor='tab:orange', markeredgecolor='white',
    markersize=8, linewidth=1.5,
    label='Pursuit-time (ΔP)'
)

# Intersection dot
opt_N = 3
opt_val = delta_S[N == opt_N][0]
ax.scatter(opt_N, opt_val, color='red', s=100, zorder=5)

# Dotted vertical from intersection
ax.vlines(opt_N, 0, opt_val, colors='grey', linestyles=':', linewidth=1)

# Checkered grid
ax.grid(True, linestyle=':', color='grey', alpha=0.7)

# N* annotation
ax.annotate(
    r'$N^* = 3$',
    xy=(opt_N, 0), xycoords=('data','axes fraction'),
    xytext=(0, -25), textcoords='offset points',
    ha='center', va='top'
)

# Labels & title (defaults)
ax.set_xlabel(r'$N$')
ax.set_ylabel('Marginal time change per prey item')
ax.set_title('Optimal Diet Choice', pad=12)

# Axis limits & ticks
ax.set_xticks(N)
ax.set_ylim(0, 0.55)

# Only left & bottom spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend (default font size)
legend = ax.legend(
    handles=[l1, l2],
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='grey',
    handlelength=1.5,
    handletextpad=0.5
)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.show()













