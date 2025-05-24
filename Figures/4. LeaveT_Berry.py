import numpy as np
import matplotlib.pyplot as plt

# ——— Berry task parameters ———
# Precomputed BRR for berry-rich environment (from fixed-point)
brr_berry_rich = 23.7513

# Patch initial yields: low and high
s_vals = [34.5, 57.5]
colors = ['tab:blue', 'tab:green']  # low=blue, high=green

# Decay rate for berries
lam_berry = 0.11
def g_berry(t, s):
    return s * np.exp(-lam_berry * t)

# Softmax leave‑probability (α=1 risk-sensitive)
def p_leave(t, s, beta, c, alpha, brr):
    return 1.0 / (1.0 + np.exp(c + beta * (g_berry(t, s) - alpha * brr)))

# Analytic expected leaving time
def expected_leave_time(beta, c, alpha, s, brr, N=500):
    pL = np.array([p_leave(t, s, beta, c, alpha, brr) for t in range(1, N+1)])
    surv = np.concatenate(([1.0], np.cumprod(1 - pL)[:-1]))
    p_t = pL * surv
    times = np.arange(1, N+1)
    return np.dot(times, p_t)

alpha_fixed = 1.0
N = 500

# ——— Plot A: E[T] vs β (c = -2) ———
c_fixed = -2.0
beta_range = np.linspace(0, 1, 200)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta, c_fixed, alpha_fixed, s, brr_berry_rich, N)
          for beta in beta_range]
    plt.plot(beta_range, Es, color=color)
# annotate c
E_high_at1 = expected_leave_time(1.0, c_fixed, alpha_fixed, s_vals[1], brr_berry_rich, N)
plt.text(1.02, E_high_at1, 'c = -2', color=colors[1], va='center')

plt.xlim(0, 1)
plt.xlabel('β (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Berry Task: β on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ——— Plot B: E[T] vs c (β = 0.5) ———
beta_fixed = 0.5
c_range = np.arange(-2, 4.1, 0.5)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta_fixed, c, alpha_fixed, s, brr_berry_rich, N)
          for c in c_range]
    plt.plot(c_range, Es, color=color)
# annotate β
E_high_at4 = expected_leave_time(beta_fixed, 4.0, alpha_fixed, s_vals[1], brr_berry_rich, N)
plt.text(4.2, E_high_at4, 'β = 0.5', color=colors[1], va='center')

plt.xlim(-2, 4)
plt.xticks([-2, 0, 2, 4])
plt.xlabel('c (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Berry Task: c on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ——— Plot C: E[T] vs c (β = 0, smooth) ———
beta_zero = 0.0
c_smooth = np.linspace(0, 4, 300)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta_zero, c, alpha_fixed, s, brr_berry_rich, N)
          for c in c_smooth]
    plt.plot(c_smooth, Es, color=color)
# annotate β = 0
E_high_at4_b0 = expected_leave_time(beta_zero, 4.0, alpha_fixed, s_vals[1], brr_berry_rich, N)
plt.text(4.2, E_high_at4_b0, 'β = 0', color=colors[1], va='center')

plt.xlim(0, 4)
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('c (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Berry Task: c on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
