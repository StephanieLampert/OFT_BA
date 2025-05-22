# Le Heron
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1) Task parameters
S       = np.array([32.5, 45.0, 57.5])         # initial patch yields
p_rich  = np.array([0.2, 0.3, 0.5])
p_poor  = np.array([0.5, 0.3, 0.2])
lam     = 0.075
tau     = 6.0
n_sub   = 39
trials  = 10
noise   = 1.0

# 2) Helper: compute BRR via fixed‐point (so we can print it)
def compute_brr(s_vals, p_env, lam, tau):
    def fp(brr):
        T_i      = -np.log(brr/s_vals) / lam
        integral = (s_vals/lam) * (1 - np.exp(-lam * T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr
    # bracket so that 0 < BRR <= min(s_vals)
    return brentq(fp, 1e-12, s_vals.min())

BRR_rich = compute_brr(S, p_rich, lam, tau)
BRR_poor = compute_brr(S, p_poor, lam, tau)
print(f"Rich‐env BRR = {BRR_rich:.4f}  |  Poor‐env BRR = {BRR_poor:.4f}")

# 3) Compute MVT‐optimal leaving times for each patch
def compute_Topt(S, p_env):
    def fp(brr):
        T_i      = -np.log(brr/S) / lam
        integral = (S/lam) * (1 - np.exp(-lam * T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr

    brr = brentq(fp, 1e-12, S.min())
    return -np.log(brr / S) / lam

T_opt_rich = compute_Topt(S, p_rich)
T_opt_poor = compute_Topt(S, p_poor)

# 4) Environment‐specific over‐stay biases from Le Heron et al.
bias_rich = 5.9   # on average stayed 5.9 s past optimal in rich
bias_poor = 7.1   #                               7.1 s in poor

# 5) Simulate subject means (adding noise)
rng = np.random.default_rng(0)
def simulate(T_opt, bias):
    # for each patch, add bias + small Gaussian noise averaged over trials
    return np.array([
        (T_opt[i] + bias) + rng.normal(0, noise, trials).mean()
        for i in range(len(S))
    ])

subs_rich = np.vstack([simulate(T_opt_rich, bias_rich) for _ in range(n_sub)])
subs_poor = np.vstack([simulate(T_opt_poor, bias_poor) for _ in range(n_sub)])

# 6) Group means & SEM
gm_rich = subs_rich.mean(axis=0)
sem_rich = subs_rich.std(axis=0, ddof=1) / np.sqrt(n_sub)
gm_poor = subs_poor.mean(axis=0)
sem_poor = subs_poor.std(axis=0, ddof=1) / np.sqrt(n_sub)

# 7) Plot
patches = ['Low', 'Mid', 'High']
x = np.arange(len(S))

plt.figure(figsize=(8,6))
# Subjects
plt.errorbar(x, gm_rich, yerr=sem_rich, fmt='o-', color='tab:blue',
             lw=2, capsize=4, label='Rich (subjects)')
plt.errorbar(x, gm_poor, yerr=sem_poor, fmt='s-', color='tab:green',
             lw=2, capsize=4, label='Poor (subjects)')
# MVT optima
plt.plot(x, T_opt_rich, '--', color='tab:blue', lw=1.5, label='MVT Rich')
plt.plot(x, T_opt_poor, '-.', color='tab:green', lw=1.5, label='MVT Poor')

plt.xticks(x, patches)
plt.xlabel('Patch yield')
plt.ylabel('Time in patch (s)')
plt.title('Milk Task: MVT Suboptimality')
plt.legend()
plt.tight_layout()
plt.show()


# Contreras-Huerta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# 1) Berry task parameters
S        = np.array([34.5, 57.5])      # low, high yields
p_mix    = np.array([0.5, 0.5])        # equal mix
lam      = 0.11
tau_rich = 2.8
tau_poor = 5.0

# 2) MVT fixed-point for BRR and Topt
def compute_Topt(S, p_env, lam, tau):
    def fp(brr):
        T    = -np.log(brr/S)/lam
        integral = (S/lam)*(1 - np.exp(-lam*T))
        avg_rate = p_env.dot(integral) / p_env.dot(T + tau)
        return avg_rate - brr
    brr = brentq(fp, 1e-12, S.max())
    return -np.log(brr/S)/lam

Topt_rich = compute_Topt(S, p_mix, lam, tau_rich)
Topt_poor = compute_Topt(S, p_mix, lam, tau_poor)

# 3) Over-stay biases from Contreras-Huerta SI
bias_rich = 5.30   # same for low & high in rich
bias_poor = 5.49   # average over low/high in poor

# 4) “Subject” means = optimal + bias
Tsubj_rich = Topt_rich + bias_rich
Tsubj_poor = Topt_poor + bias_poor

# 5) (Optional) simulate N subjects with per-trial noise
n_subjects     = 29
trials_per_patch = 10
sigma_noise    = 1.0
rng            = np.random.default_rng(0)

def simulate_subject(Tsubj):
    # each subject: per-patch average over a few noisy trials
    return np.array([
        Tsubj[i] + rng.normal(0, sigma_noise, trials_per_patch).mean()
        for i in range(len(S))
    ])

subs_rich = np.vstack([simulate_subject(Tsubj_rich) for _ in range(n_subjects)])
subs_poor = np.vstack([simulate_subject(Tsubj_poor) for _ in range(n_subjects)])

gm_rich = subs_rich.mean(axis=0)
sem_rich = subs_rich.std(axis=0, ddof=1)/np.sqrt(n_subjects)
gm_poor = subs_poor.mean(axis=0)
sem_poor = subs_poor.std(axis=0, ddof=1)/np.sqrt(n_subjects)

# 6) Plot
patch_labels = ['Low','High']
x = np.arange(len(S))

plt.figure(figsize=(6,4))
# Simulated “subjects”
plt.errorbar(x, gm_rich, yerr=sem_rich, fmt='o-', color='tab:blue',
             capsize=4, lw=2, label='Rich (subjects)')
plt.errorbar(x, gm_poor, yerr=sem_poor, fmt='s-', color='tab:green',
             capsize=4, lw=2, label='Poor (subjects)')
# MVT curves
plt.plot(x, Topt_rich, '--', color='tab:blue', label='MVT (Rich)')
plt.plot(x, Topt_poor, '-.', color='tab:green', label='MVT (Poor)')

plt.xticks(x, patch_labels)
plt.xlabel('Patch yield')
plt.ylabel('Time in patch (s)')
plt.title('Berry Task: MVT Suboptimality')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


