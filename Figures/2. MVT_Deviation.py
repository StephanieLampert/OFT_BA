import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1) Common styling: same figure size and font sizes for all text
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# ——————————————————————————————————————————————————————————————
# Milk Task: MVT Suboptimality
# ——————————————————————————————————————————————————————————————

# Task parameters
S       = np.array([32.5, 45.0, 57.5])
p_rich  = np.array([0.2, 0.3, 0.5])
p_poor  = np.array([0.5, 0.3, 0.2])
lam     = 0.075
tau     = 6.0
n_sub   = 39
trials  = 10
noise   = 1.0

def compute_brr(s_vals, p_env, lam, tau):
    def fp(brr):
        T_i      = -np.log(brr/s_vals) / lam
        integral = (s_vals/lam) * (1 - np.exp(-lam * T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr
    return brentq(fp, 1e-12, s_vals.min())

def compute_Topt(S, p_env):
    def fp(brr):
        T_i      = -np.log(brr/S) / lam
        integral = (S/lam) * (1 - np.exp(-lam * T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr
    brr = brentq(fp, 1e-12, S.min())
    return -np.log(brr / S) / lam

BRR_rich    = compute_brr(S, p_rich, lam, tau)
BRR_poor    = compute_brr(S, p_poor, lam, tau)
T_opt_rich  = compute_Topt(S, p_rich)
T_opt_poor  = compute_Topt(S, p_poor)

bias_rich = 5.9
bias_poor = 7.1

rng = np.random.default_rng(0)

def simulate(T_opt, bias):
    return np.array([
        (T_opt[i] + bias) + rng.normal(0, noise, trials).mean()
        for i in range(len(S))
    ])

# simulate subject data
subs_rich = np.vstack([simulate(T_opt_rich, bias_rich) for _ in range(n_sub)])
subs_poor = np.vstack([simulate(T_opt_poor, bias_poor) for _ in range(n_sub)])

# compute means and SEMs
gm_rich  = subs_rich.mean(axis=0)
sem_rich = subs_rich.std(axis=0, ddof=1) / np.sqrt(n_sub)
gm_poor  = subs_poor.mean(axis=0)
sem_poor = subs_poor.std(axis=0, ddof=1) / np.sqrt(n_sub)

patches = ['Low', 'Mid', 'High']
x = np.arange(len(S))

# plot Milk task
plt.figure()
plt.errorbar(x, gm_rich, yerr=sem_rich,
             fmt='o-', color='tab:blue', lw=2, capsize=4,
             label='Rich (subjects)')
plt.errorbar(x, gm_poor, yerr=sem_poor,
             fmt='s-', color='tab:green', lw=2, capsize=4,
             label='Poor (subjects)')
plt.plot(x, T_opt_rich, '--', color='tab:blue', lw=1.5, label='MVT Rich')
plt.plot(x, T_opt_poor, '-.', color='tab:green', lw=1.5, label='MVT Poor')

plt.xticks(x, patches)
plt.xlabel('Patch yield')
plt.ylabel('Time in patch (s)')
plt.title('Milk Task: MVT Suboptimality')
plt.legend()
plt.tight_layout()
plt.show()


# ——————————————————————————————————————————————————————————————
# Berry Task: MVT Suboptimality
# ——————————————————————————————————————————————————————————————

S        = np.array([34.5, 57.5])
p_mix    = np.array([0.5, 0.5])
lam      = 0.11
tau_rich = 2.8
tau_poor = 5.0

def compute_Topt_berry(S, p_env, lam, tau):
    def fp(brr):
        T        = -np.log(brr/S) / lam
        integral = (S/lam) * (1 - np.exp(-lam * T))
        avg_rate = p_env.dot(integral) / p_env.dot(T + tau)
        return avg_rate - brr
    brr = brentq(fp, 1e-12, S.max())
    return -np.log(brr / S) / lam

Topt_rich_b = compute_Topt_berry(S, p_mix, lam, tau_rich)
Topt_poor_b = compute_Topt_berry(S, p_mix, lam, tau_poor)

bias_rich_b = 5.30
bias_poor_b = 5.49

# simulate Berry subjects using the same simulate(T_opt, bias) function
subs_rich_b = np.vstack([
    simulate(Topt_rich_b, bias_rich_b) for _ in range(n_sub)
])
subs_poor_b = np.vstack([
    simulate(Topt_poor_b, bias_poor_b) for _ in range(n_sub)
])

gm_rich_b  = subs_rich_b.mean(axis=0)
sem_rich_b = subs_rich_b.std(axis=0, ddof=1) / np.sqrt(n_sub)
gm_poor_b  = subs_poor_b.mean(axis=0)
sem_poor_b = subs_poor_b.std(axis=0, ddof=1) / np.sqrt(n_sub)

patch_labels = ['Low', 'High']
x2 = np.arange(len(S))

plt.figure()
plt.errorbar(x2, gm_rich_b, yerr=sem_rich_b,
             fmt='o-', color='tab:blue', lw=2, capsize=4,
             label='Rich (subjects)')
plt.errorbar(x2, gm_poor_b, yerr=sem_poor_b,
             fmt='s-', color='tab:green', lw=2, capsize=4,
             label='Poor (subjects)')
plt.plot(x2, Topt_rich_b, '--', color='tab:blue', lw=1.5, label='MVT (Rich)')
plt.plot(x2, Topt_poor_b, '-.', color='tab:green', lw=1.5, label='MVT (Poor)')

plt.xticks(x2, patch_labels)
plt.xlabel('Patch yield')
plt.ylabel('Time in patch (s)')
plt.title('Berry Task: MVT Suboptimality')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()






