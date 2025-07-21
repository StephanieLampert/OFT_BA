import numpy as np
import pandas as pd

# ----------------------------------------
# 1) Helper Functions
# ----------------------------------------
def simulate_patch(alpha, beta, s_initial, lam_decay, c0, K, max_time=200):
    """
    Simulates a single patch trial under the stochastic softmax rule.
    Returns the time t at which the forager leaves the patch.
    """
    c_eff = c0 + K * (1 - alpha)
    for t in range(1, max_time + 1):
        # Current reward
        g_t = s_initial * np.exp(-lam_decay * t)
        if g_t <= 1e-12:
            return t

        # Softmax leave‐probability
        logit = c_eff + beta * g_t
        if logit > 500:
            p_leave = 0.0
        elif logit < -500:
            p_leave = 1.0
        else:
            p_leave = 1.0 / (1.0 + np.exp(logit))

        # Draw a random decision
        if np.random.rand() < p_leave:
            return t

    return max_time


def get_leaving_probabilities_and_survival(c_eff, beta, s_initial, time_steps, lam_decay):
    """
    Returns two arrays over the vector time_steps:
      - p_leave:     p(leave | t)
      - p_survive:   probability of still being in the patch at time t (i.e. not having left before t)
    """
    # 1) reward schedule
    rewards = s_initial * np.exp(-lam_decay * time_steps)
    rewards = np.maximum(rewards, 1e-9)

    # 2) conditional leave probability
    logits = c_eff + beta * rewards
    logits = np.clip(logits, -500, 500)
    p_leave = 1.0 / (1.0 + np.exp(logits))

    # 3) survival up to time t
    p_stay = 1.0 - p_leave
    p_stay = np.minimum(p_stay, 1.0 - 1e-9)
    p_survive = np.concatenate(([1.0], np.cumprod(p_stay)))[:-1]

    return p_leave, p_survive


def calculate_expected_leaving_time(p_leave, p_survive, time_steps):
    """
    Analytically computes E[T] = sum_t t * p(leave exactly at t),
    where p(leave exactly at t) = p_leave[t] * p_survive[t].
    """
    p_exact = p_leave * p_survive
    return np.sum(time_steps * p_exact)


# ----------------------------------------
# 2) Simulation Parameters
# ----------------------------------------
PATCH_YIELDS = np.array([32.5, 45.0, 57.5])
LAMBDA_DECAY = 0.075
C0 = -4.0
K = 6.0
BETA = {'Rich': 0.4918, 'Poor': 0.5338}
ALPHAS = [0.5, 1.5]
N_MC = 1000
MAX_TIME = 200
TIME_STEPS = np.arange(1, MAX_TIME + 1)

# ----------------------------------------
# 3) Run Analytic vs Monte Carlo Comparison
# ----------------------------------------
rows = []
for alpha in ALPHAS:
    c_eff = C0 + K*(1 - alpha)
    for env_label, beta in BETA.items():
        for s in PATCH_YIELDS:
            # Analytic E[T]
            p_leave, p_survive = get_leaving_probabilities_and_survival(
                c_eff, beta, s, TIME_STEPS, LAMBDA_DECAY
            )
            E_analytic = calculate_expected_leaving_time(p_leave, p_survive, TIME_STEPS)

            # Monte Carlo
            sims = np.array([
                simulate_patch(alpha, beta, s, LAMBDA_DECAY, C0, K, MAX_TIME)
                for _ in range(N_MC)
            ])
            E_sim = sims.mean()
            STD_sim = sims.std(ddof=1)
            diff = E_sim - E_analytic

            rows.append({
                'α': alpha,
                'Env': env_label,
                'Yield': f"{s:.1f}",
                'E[T]_analytic (s)': f"{E_analytic:.2f}",
                'E[T]_sim (s)':      f"{E_sim:.2f}",
                'Δ (s)':             f"{diff:.2f}",
                'Std (s)':           f"{STD_sim:.2f}"
            })

df = pd.DataFrame(rows)

# Print results table
print(df.to_string(index=False))

