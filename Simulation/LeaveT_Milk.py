import numpy as np
import pandas as pd
from scipy.special import expit

# 1) Task parameters
Y = np.array([32.5, 45.0, 57.5])       # Patch yields
lam = 0.075                            # Decay rate
BRR_rich = 21.8710                     # Fixed BRR value for rich env
BRR_poor = 18.5684                     # Fixed BRR value for poor env
alpha = 1.0                            # Risk parameter
Tmax = 200                             # Time range for calculations

# 2) Softmax-based expected time & variance
def analytic_moments(Y, lam, BRR, beta, c, alpha=1.0, Tmax=200):
    t = np.arange(1, Tmax + 1)
    g = Y[:, None] * np.exp(-lam * t[None, :])
    z = c + beta * (g - alpha * BRR)
    p = expit(z)
    surv = np.cumprod(1 - p, axis=1)
    shift = np.hstack([np.ones((len(Y), 1)), surv[:, :-1]])
    fe = p * shift
    E = np.sum(t * fe, axis=1)
    Var = np.sum((t - E[:, None]) ** 2 * fe, axis=1)
    return E, Var

# 3) Parameter grid
beta_vals = np.logspace(-2, 1, 5)   # [0.01, 0.1, 1, 3.16, 10]
c_vals = np.linspace(-4, 2, 5)      # [-4, -2.5, -1, 0.5, 2]

# 4) Loop and compute
rows = []
for beta in beta_vals:
    for c in c_vals:
        E_rich, V_rich = analytic_moments(Y, lam, BRR_rich, beta, c, alpha, Tmax)
        E_poor, V_poor = analytic_moments(Y, lam, BRR_poor, beta, c, alpha, Tmax)
        for i, y in enumerate(Y):
            rows.append({
                'Yield': y,
                'Beta': beta,
                'c': c,
                'Env': 'rich',
                'E[T]': E_rich[i],
                'Var[T]': V_rich[i]
            })
            rows.append({
                'Yield': y,
                'Beta': beta,
                'c': c,
                'Env': 'poor',
                'E[T]': E_poor[i],
                'Var[T]': V_poor[i]
            })

# 5) Create and show/save DataFrame
df_final = pd.DataFrame(rows)

# Print first few rows
print(df_final.head())

# Optionally save to CSV
df_final.to_csv("milk_task_results.csv", index=False)

