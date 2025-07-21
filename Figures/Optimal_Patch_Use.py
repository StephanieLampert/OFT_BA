import matplotlib.pyplot as plt
import numpy as np

# Data from N=0 to N=6
N = np.arange(0, 7)
delta_T = np.array([0.50, 0.17, 0.08, 0.05, 0.03, 0.02, 0.01])  # Travel-time (ΔT)
delta_H = np.array([0.01, 0.03, 0.05, 0.05, 0.06, 0.07, 0.08])  # Hunting-time (ΔH)

fig, ax = plt.subplots(figsize=(6, 4))

# Draw grid behind curves
ax.set_axisbelow(True)
ax.grid(True, linestyle=':', color='grey', alpha=0.7)

# Plot Travel-time: blue upward triangles
l1, = ax.plot(
    N, delta_T,
    marker='^', color='tab:blue',
    markerfacecolor='tab:blue', markeredgecolor='white',
    markersize=8, linewidth=1.5,
    label='Travel-time (ΔT)'
)

# Plot Hunting-time: orange downward triangles
l2, = ax.plot(
    N, delta_H,
    marker='v', color='tab:orange',
    markerfacecolor='tab:orange', markeredgecolor='white',
    markersize=8, linewidth=1.5,
    label='Hunting-time (ΔH)'
)

# Intersection at N* = 3
opt_N = 3
opt_val = delta_T[N == opt_N][0]

# Dotted vertical drop from x=3 down to the x-axis
ax.vlines(opt_N, 0, opt_val,
          colors='grey', linestyles=':', linewidth=1, zorder=1)

# Red dot on top
ax.scatter(opt_N, opt_val,
           marker='o',
           color='red',
           edgecolors='none',
           s=100,
           zorder=5)

# Annotate N* under the '3' tick, reduced offset
ax.annotate(
    r'$N^* = 3$',
    xy=(opt_N, 0), xycoords=('data','axes fraction'),
    xytext=(0, -20),          # reduced to 20 points
    textcoords='offset points',
    ha='center', va='top'
)

# Axis label N with smaller pad
ax.set_xlabel(r'$N$', labelpad=15)

# Slightly adjust tick label padding
ax.tick_params(axis='x', which='major', pad=5)

# Remaining labels and title
ax.set_ylabel('Marginal time change per patch type')
ax.set_title('Optimal Patch Use', pad=12)

# Ticks and limits
ax.set_xticks(N)
ax.set_xlim(0, 6)
ax.set_ylim(0, 0.55)

# Only bottom & left spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Framed square legend
legend = ax.legend(
    handles=[l1, l2],
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='grey',
    handlelength=1.5,
    handletextpad=0.5,
    fontsize=10
)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.show()



