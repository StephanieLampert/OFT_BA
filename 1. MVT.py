# Le Heron

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1. Decay function
def g(x, s, lam=0.075):
    return s * np.exp(-lam * x)

# 2. Patch yields and labels
s_vals = np.array([32.5, 45.0, 57.5])
labels = ['Low (32.5)', 'Mid (45.0)', 'High (57.5)']

# 3. Curve colors
curve_colors = ['tab:blue', 'tab:orange', 'tab:green']

# 4. Environment proportions and parameters
p_rich = np.array([0.2, 0.3, 0.5])
p_poor = np.array([0.5, 0.3, 0.2])
lam = 0.075
tau = 6.0

# 5. Compute BRR via fixed-point
def compute_brr(s_vals, p_env, lam, tau):
    def fp(brr):
        T_i = -np.log(brr / s_vals) / lam
        integral = (s_vals / lam) * (1 - np.exp(-lam * T_i))
        avg_rate = np.dot(p_env, integral) / np.dot(p_env, (T_i + tau))
        return avg_rate - brr
    return brentq(fp, 1e-6, s_vals.max())

rich_rate = compute_brr(s_vals, p_rich, lam, tau)
poor_rate = compute_brr(s_vals, p_poor, lam, tau)
env_rates = {'rich': rich_rate, 'poor': poor_rate}
env_colors = {'rich': 'gold', 'poor': 'green'}

# 6. Compute intersection points
intersection_points = []
for env, rate in env_rates.items():
    for idx, s_val in enumerate(s_vals):
        if rate <= s_val:
            x_cross = -np.log(rate / s_val) / lam
            y_cross = g(x_cross, s_val, lam)
            intersection_points.append((x_cross, y_cross, idx))

# 7. Plotting
fig, ax = plt.subplots(figsize=(8, 5))

# a) Decay curves
x = np.linspace(0, 100, 400)
for idx, (s_val, label) in enumerate(zip(s_vals, labels)):
    ax.plot(x, g(x, s_val, lam), label=label, color=curve_colors[idx], linewidth=2)

# b) Environment lines
for env, rate in env_rates.items():
    ax.hlines(rate, xmin=0, xmax=100,
              colors=env_colors[env],
              linestyles='--',
              label=f"{env.capitalize()} rate")

# c) Intersection markers
for x_cross, y_cross, idx in intersection_points:
    ax.scatter(x_cross, y_cross,
               color=curve_colors[idx],
               marker='o', s=50, zorder=5)

# d) Axes formatting
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_aspect('equal', adjustable='box')

# e) Labels, title, legend, grid
ax.set_xlabel('Time in patch $t$')
ax.set_ylabel('Patch reward rate $g(t)$')
ax.set_title('Milk Task')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.5)

plt.show()

#Contreras-Huerta

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1. Decay function for berry task
def g(x, s, lam=0.11):
    return s * np.exp(-lam * x)

# 2. Patch yields for berry task
s_vals = np.array([34.5, 57.5])
labels = ['Low (34.5)', 'High (57.5)']

# 3. Colors: low curve = blue, high curve = green
curve_colors = ['tab:blue', 'tab:green']

# 4. Environment parameters
p = np.array([0.5, 0.5])  # patch mix
lam = 0.11
tau_rich = 2.8
tau_poor = 5.0

# 5. Compute BRR via fixed-point
def compute_brr(s_vals, p_env, lam, tau):
    def fp(brr):
        T_i = -np.log(brr / s_vals) / lam
        integral = (s_vals / lam) * (1 - np.exp(-lam * T_i))
        avg_rate = np.dot(p_env, integral) / np.dot(p_env, (T_i + tau))
        return avg_rate - brr
    return brentq(fp, 1e-6, s_vals.max())

rich_rate = compute_brr(s_vals, p, lam, tau_rich)
poor_rate = compute_brr(s_vals, p, lam, tau_poor)
env_rates = {'rich': rich_rate, 'poor': poor_rate}
env_colors = {'rich': 'gold', 'poor': 'green'}

# 6. Compute intersection points
intersection_points = []
for env, rate in env_rates.items():
    for idx, s_val in enumerate(s_vals):
        if rate <= s_val:
            x_cross = -np.log(rate / s_val) / lam
            y_cross = g(x_cross, s_val, lam)
            intersection_points.append((x_cross, y_cross, idx))

# 7. Plotting
fig, ax = plt.subplots(figsize=(8, 5))

# a) Decay curves
x = np.linspace(0, 100, 400)
for idx, (s_val, label) in enumerate(zip(s_vals, labels)):
    ax.plot(x, g(x, s_val, lam), label=label, color=curve_colors[idx], linewidth=2)

# b) Environment lines
for env, rate in env_rates.items():
    ax.hlines(rate, xmin=0, xmax=100,
              colors=env_colors[env], linestyles='--',
              label=f"{env.capitalize()} rate")

# c) Intersection markers (dots)
for x_cross, y_cross, idx in intersection_points:
    ax.scatter(x_cross, y_cross,
               color=curve_colors[idx], marker='o', s=50, zorder=5)

# d) Axis formatting
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_aspect('equal', adjustable='box')

# e) Labels, title, legend, grid
ax.set_xlabel('Time in patch t')
ax.set_ylabel('Patch reward rate g(t)')
ax.set_title('Berry Task')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.5)

plt.show()
















