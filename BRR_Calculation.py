#Le-Heron
import numpy as np
from scipy.optimize import brentq

# 1) Task parameters
s_vals = np.array([32.5, 45.0, 57.5])
lam    = 0.075
tau    = 6.0
p_rich = np.array([0.2, 0.3, 0.5])
p_poor = np.array([0.5, 0.3, 0.2])

# 2) Fixed‐point solver
def compute_brr(s, p_env, lam, tau):
    def fp(brr):
        T_i = -np.log(brr/s) / lam
        integral = (s/lam) * (1 - np.exp(-lam*T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr
    # bracket from near zero up to the largest s_i
    return brentq(fp, 1e-12, s.max())

rich_brr = compute_brr(s_vals, p_rich, lam, tau)
poor_brr = compute_brr(s_vals, p_poor, lam, tau)

print(f"Rich‐env BRR ≈ {rich_brr:.4f}")
print(f"Poor‐env BRR ≈ {poor_brr:.4f}")

#Contreras-Huerta
import numpy as np
from scipy.optimize import brentq

# Berry task parameters
s_vals = np.array([34.5, 57.5])
lam    = 0.11
tau_rich = 2.8
tau_poor = 5.0
p_mix = np.array([0.5, 0.5])   # same in rich & poor

def compute_brr(s_vals, p_env, lam, tau):
    def fp(brr):
        T_i      = -np.log(brr/s_vals) / lam
        integral = (s_vals/lam) * (1 - np.exp(-lam * T_i))
        avg_rate = p_env.dot(integral) / p_env.dot(T_i + tau)
        return avg_rate - brr

    # bracket so that 0 < BRR <= min(s_vals)
    return brentq(fp, 1e-12, s_vals.min())

rich_brr = compute_brr(s_vals, p_mix, lam, tau_rich)
poor_brr = compute_brr(s_vals, p_mix, lam, tau_poor)

print(f"Berry—Rich BRR ≃ {rich_brr:.4f}")
print(f"Berry—Poor BRR ≃ {poor_brr:.4f}")




