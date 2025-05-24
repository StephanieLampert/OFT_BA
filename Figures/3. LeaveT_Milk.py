import numpy as np
import matplotlib.pyplot as plt

# Precomputed BRR for milk-rich environment
BRR_RICH = 21.8710

# Patch initial yields and colors
s_vals = [32.5, 45.0, 57.5]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

# Exponential decay function
decay_rate = 0.075
def reward_exp(t, s):
    return s * np.exp(-decay_rate * t)

# Softmax leave-probability function
def p_leave(t, s, beta, c, alpha, brr):
    """Probability to leave at time t for patch with initial s."""
    return 1.0 / (1.0 + np.exp(c + beta * (reward_exp(t, s) - alpha * brr)))

# Analytical expected leaving time
def expected_leave_time(beta, c, alpha, s, brr, N=500):
    """Compute E[T] = sum_{t=1}^N t * Pr(leave at t)."""
    pL = np.array([p_leave(t, s, beta, c, alpha, brr) for t in range(1, N+1)])
    surv = np.concatenate(([1.0], np.cumprod(1 - pL)[:-1]))
    p_t = pL * surv
    times = np.arange(1, N+1)
    return np.dot(times, p_t)

alpha_fixed = 1.0
N = 500

# 1) Plot: E[T] vs β (c = -2)
c_fixed = -2.0
beta_range = np.linspace(0, 1, 200)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta, c_fixed, alpha_fixed, s, BRR_RICH, N)
          for beta in beta_range]
    plt.plot(beta_range, Es, color=color)

# Annotation next to the green curve at β=1
E_high_beta1 = expected_leave_time(1.0, c_fixed, alpha_fixed, s_vals[2], BRR_RICH, N)
plt.text(1.02, E_high_beta1, 'c = -2', color=colors[2], va='center')

plt.xlim(0, 1)
plt.xlabel('β (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Milk Task: β on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 2) Plot: E[T] vs c (β = 0.5)
beta_fixed = 0.5
c_range = np.arange(-2, 4.1, 0.5)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta_fixed, c, alpha_fixed, s, BRR_RICH, N)
          for c in c_range]
    plt.plot(c_range, Es, color=color)

# Annotation next to the green curve at c=4
E_high_c4 = expected_leave_time(beta_fixed, 4.0, alpha_fixed, s_vals[2], BRR_RICH, N)
plt.text(4.2, E_high_c4, 'β = 0.5', color=colors[2], va='center')

plt.xlim(-2, 4)
plt.xticks([-2, 0, 2, 4])
plt.xlabel('c (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Milk Task: c on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# 3) Plot: E[T] vs c (β = 0, smooth)
beta_zero = 0.0
c_smooth = np.linspace(0, 4, 300)

plt.figure(figsize=(6, 4))
for s, color in zip(s_vals, colors):
    Es = [expected_leave_time(beta_zero, c, alpha_fixed, s, BRR_RICH, N)
          for c in c_smooth]
    plt.plot(c_smooth, Es, color=color)

# Annotation next to the green curve at c=4
E_high_c4_beta0 = expected_leave_time(beta_zero, 4.0, alpha_fixed, s_vals[2], BRR_RICH, N)
plt.text(4.2, E_high_c4_beta0, 'β = 0', color=colors[2], va='center')

plt.xlim(0, 4)
plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('c (higher = exploit)')
plt.ylabel('Expected Leaving Time (s)')
plt.title('Milk Task: c on E(t)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




