import numpy as np
import pandas as pd
from scipy.special import expit

# 1) Berry task parameters
Y = np.array([34.5, 57.5])         # Patch yields
lam = 0.11                         # Decay rate
BRR_rich = 23.7513                 # Provided rich BRR
BRR_poor = 19.2604                 # Provided poor BRR
alpha = 1.0                        # Risk parameter
Tmax = 200                         # Max patch time to simulate

# 2) Custom softmax model: your version
def analytic_moments_custom_softmax(Y, lam, BRR, beta, c, alpha=1.0, Tmax=200):
    t = np.arange(1, Tmax + 1)
    g = Y[:, None] * np.exp(-lam * t[None, :])
    z = c + beta * (g - alpha * BRR)  # Your softmax formulation
    p = expit(z)
    surv = np.cumprod(1 - p, axis=1)
    shift = np.hstack([np.ones((len(Y), 1)), surv[:, :-1]])
    fe = p * shift
    E = np.sum(t * fe, axis=1)
    Var = np.sum((t - E[:, None]) ** 2 * fe, axis=1)
    return E, Var

# 3) Parameter grid
beta_vals = np.logspace(-2, 1, 5)  # [0.01, 0.1, 1.0, 3.16, 10.0]
c_vals = np.linspace(-4, 2, 5)     # [-4, -2.5, -1, 0.5, 2.0]

# 4) Loop through grid and calculate E[T], Var[T]
rows = []
for beta in beta_vals:
    for c in c_vals:
        E_rich, V_rich = analytic_moments_custom_softmax(Y, lam, BRR_rich, beta, c, alpha, Tmax)
        E_poor, V_poor = analytic_moments_custom_softmax(Y, lam, BRR_poor, beta, c, alpha, Tmax)
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

# 5) Create DataFrame and output
df_berry = pd.DataFrame(rows)

# Print preview
print(df_berry.head())

# Optionally save to CSV
df_berry.to_csv("berry_task_results.csv", index=False)
