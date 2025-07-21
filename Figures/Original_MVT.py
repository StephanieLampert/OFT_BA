import numpy as np
import matplotlib.pyplot as plt

# 1. Define the patch gain functions and optimal slope ----------------------
def g_high(T):
    return 100 * (1 - np.exp(-0.1 * T))

def g_low(T):
    return  60 * (1 - np.exp(-0.1 * T))

T = np.linspace(0, 60, 400)
E_opt = 1.2   # optimal net intake rate (slope)

# 2. Solve for T* such that g'(T*) = E_opt ------------------------------------
T_high_opt = -10 * np.log(E_opt / 10)
T_low_opt  = -10 * np.log(E_opt /  6)

# 3. Build tangent lines at T* -----------------------------------------------
y_high_tan = E_opt * (T - T_high_opt) + g_high(T_high_opt)
y_low_tan  = E_opt * (T - T_low_opt)  + g_low(T_low_opt)

# 4. Plot, with unified styling ----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Main gain curves
ax.plot(T, g_high(T), color='#ff7f0e', lw=2, label='High-yield patch')
ax.plot(T, g_low(T),  color='#1f77b4', lw=2, label='Low-yield patch')

# Tangent lines
ax.plot(T, y_high_tan,    color='#ff7f0e', ls='--', lw=1.5)
ax.plot(T, y_low_tan,     color='#1f77b4', ls='--', lw=1.5)

# Optimal points
ax.scatter([T_high_opt], [g_high(T_high_opt)], color='#ff7f0e', s=50, zorder=5)
ax.scatter([T_low_opt ], [g_low(T_low_opt )], color='#1f77b4', s=50, zorder=5)
ax.text(T_high_opt, g_high(T_high_opt)+2, 'T₁*', ha='center')
ax.text(T_low_opt,  g_low(T_low_opt )+2, 'T₂*', ha='center')

# Optimal intake‐rate line
ax.plot(T, E_opt * T, color='gray', ls=':')
ax.text(10, E_opt*10 + 2, 'Eₙ*', color='gray')

# Axes through the origin
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Grid, ticks and limits: turn on both x- and y-grid
ax.grid(which='both', linestyle='--', alpha=0.3)

ax.set_xlim(0, 60)
ax.set_ylim(0, np.max(g_high(T)) + 10)
ax.set_xticks([0, 20, 40, 60])
ax.set_yticks(np.linspace(0, 100, 6))

# Labels and legend
ax.set_xlabel('Time in patch (T)')
ax.set_ylabel('Energy intake g(T)')
ax.set_title('MVT', pad=12)
ax.legend(frameon=True, loc='lower right')

plt.tight_layout()
plt.show()







