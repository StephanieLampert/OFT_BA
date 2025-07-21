import numpy as np
import matplotlib.pyplot as plt

# --- Task Parameters (Milk Task) ---
PATCH_INITIAL_YIELDS = np.array([32.5, 45.0, 57.5])
PATCH_LABELS = ['Low (32.5)', 'Mid (45.0)', 'High (57.5)']
LAMBDA_DECAY_TASK = 0.075

# --- MVT Optimal Leaving Times (Le Heron et al.) ---
T_OPT_RICH_ENV = np.array([5.28, 9.62, 12.89])
T_OPT_POOR_ENV = np.array([7.46, 11.80, 15.07])

# --- “Real Foraging Deviation” Biases ---
BIAS_RICH_ENV = 5.9
BIAS_POOR_ENV = 7.1

# --- Environment‐specific β (Milk Task) ---
# Values computed from BRR_rich ≈21.8710, BRR_poor ≈18.5684,
# with γ = –0.5, λ = 2.3:
BETA_MILK_RICH = 0.4918
BETA_MILK_POOR = 0.5338

# --- Risk‐Profile Parameters ---
C0_BASELINE = -4.0
K_RISK_SENSITIVITY = 6.0

ALPHA_PROFILES_RISK_SIM = {
    'Risk Averse (α=0.5)': 0.5,
    'Risk Seeking (α=1.5)': 1.5,
}

# --- Simulation settings ---
MAX_TIME_STEPS = 100
TIME_STEPS_ET = np.arange(1, MAX_TIME_STEPS + 1)

# --- Helper Functions ---
def calculate_reward_rate_at_time(s_initial, t, lam_decay):
    return s_initial * np.exp(-lam_decay * t)

def get_leaving_probabilities_and_survival(c_eff, beta, s_initial, time_steps_arr, lam_decay):
    current_rewards = calculate_reward_rate_at_time(s_initial, time_steps_arr, lam_decay)
    current_rewards = np.maximum(current_rewards, 1e-9)
    logits = c_eff + beta * current_rewards
    logits = np.clip(logits, -500, 500)
    prob_leave_at_n = 1 / (1 + np.exp(logits))
    prob_stay_sequence = 1.0 - prob_leave_at_n
    prob_stay_sequence = np.minimum(prob_stay_sequence, 1.0 - 1e-9)
    prob_not_left_before_t = np.concatenate(([1.0], np.cumprod(prob_stay_sequence)))[:-1]
    return prob_leave_at_n, prob_not_left_before_t

def calculate_expected_leaving_time(prob_leave_at_n, prob_not_left_before_t, time_steps_arr):
    prob_leave_exactly_at_n = prob_leave_at_n * prob_not_left_before_t
    sum_probs = np.sum(prob_leave_exactly_at_n)
    if sum_probs < 0.99 and MAX_TIME_STEPS > 10:
        pass
    expected_T = np.sum(time_steps_arr * prob_leave_exactly_at_n)
    if expected_T > MAX_TIME_STEPS * 0.98 and MAX_TIME_STEPS > 10:
        print(f"Warning: E[T] ({expected_T:.2f}s) is close to MAX_TIME_STEPS ({MAX_TIME_STEPS}s). Result might be truncated.")
    return expected_T

# --- 1. “Real Foraging Deviation” Times (MVT + Bias) ---
T_real_dev_rich = T_OPT_RICH_ENV + BIAS_RICH_ENV
T_real_dev_poor = T_OPT_POOR_ENV + BIAS_POOR_ENV

# --- 2. Calculate E[T] for Simulated Risk Profiles in Rich & Poor ---
simulated_risk_ET_rich = {}
simulated_risk_ET_poor = {}

print(f"Simulating with: C0={C0_BASELINE}, K={K_RISK_SENSITIVITY}\n")
for profile_name, alpha_val in ALPHA_PROFILES_RISK_SIM.items():
    c_eff = C0_BASELINE + K_RISK_SENSITIVITY * (1 - alpha_val)
    et_values_rich = []
    et_values_poor = []
    print(f"Profile: {profile_name} (α={alpha_val}, c_eff={c_eff:.2f})")
    for s_initial in PATCH_INITIAL_YIELDS:
        # Rich environment
        prob_leave_n_r, prob_not_left_r = get_leaving_probabilities_and_survival(
            c_eff, BETA_MILK_RICH, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK
        )
        ET_rich = calculate_expected_leaving_time(prob_leave_n_r, prob_not_left_r, TIME_STEPS_ET)
        et_values_rich.append(ET_rich)
        # Poor environment
        prob_leave_n_p, prob_not_left_p = get_leaving_probabilities_and_survival(
            c_eff, BETA_MILK_POOR, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK
        )
        ET_poor = calculate_expected_leaving_time(prob_leave_n_p, prob_not_left_p, TIME_STEPS_ET)
        et_values_poor.append(ET_poor)

        patch_idx = list(PATCH_INITIAL_YIELDS).index(s_initial)
        print(f"  Patch: {PATCH_LABELS[patch_idx]:<15} | E[T]_rich = {ET_rich:>5.2f}s | E[T]_poor = {ET_poor:>5.2f}s")

    simulated_risk_ET_rich[profile_name] = np.array(et_values_rich)
    simulated_risk_ET_poor[profile_name] = np.array(et_values_poor)
    print("-" * 60)

# --- 3. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
x_indices = np.arange(len(PATCH_LABELS))

plot_styles = {
    'MVT Optimal Rich':    {'color': 'forestgreen', 'linestyle': '-',  'marker': 'o', 'linewidth': 2,  'markersize': 7},
    'MVT Optimal Poor':    {'color': 'forestgreen', 'linestyle': '-',  'marker': 'o', 'linewidth': 2,  'markersize': 7},
    'Real Foraging Rich':  {'color': 'purple',     'linestyle': '-',  'marker': 'o', 'linewidth': 2,  'markersize': 7},
    'Real Foraging Poor':  {'color': 'purple',     'linestyle': '-',  'marker': 'o', 'linewidth': 2,  'markersize': 7},
    'Risk Averse (α=0.5)': {'color': 'blue',       'linestyle': ':',  'marker': 'o', 'linewidth': 2.5,'markersize': 9},
    'Risk Seeking (α=1.5)':{'color': 'darkorange', 'linestyle': ':',  'marker': 'o', 'linewidth': 2.5,'markersize': 9}
}

# Rich Environment Plot
ax_rich = axes[0]
ax_rich.plot(x_indices, T_OPT_RICH_ENV,    **plot_styles['MVT Optimal Rich'],    label='MVT Optimal (Rich)')
ax_rich.plot(x_indices, T_real_dev_rich,   **plot_styles['Real Foraging Rich'],  label='Real Foraging (Rich)')
for profile_name in ALPHA_PROFILES_RISK_SIM.keys():
    ax_rich.plot(x_indices, simulated_risk_ET_rich[profile_name], **plot_styles[profile_name],
                 label=f"Sim: {profile_name}")

ax_rich.set_xticks(x_indices, PATCH_LABELS)
ax_rich.set_xlabel('Patch Type (Initial Yield)')
ax_rich.set_ylabel('Time in Patch (seconds)')
ax_rich.set_title('Milk Task: Rich Environment')
ax_rich.legend(loc='best')
ax_rich.grid(True, linestyle=':', alpha=0.7)

# Poor Environment Plot
ax_poor = axes[1]
ax_poor.plot(x_indices, T_OPT_POOR_ENV,    **plot_styles['MVT Optimal Poor'],    label='MVT Optimal (Poor)')
ax_poor.plot(x_indices, T_real_dev_poor,   **plot_styles['Real Foraging Poor'],  label='Real Foraging (Poor)')
for profile_name in ALPHA_PROFILES_RISK_SIM.keys():
    ax_poor.plot(x_indices, simulated_risk_ET_poor[profile_name], **plot_styles[profile_name],
                 label=f"Sim: {profile_name}")

ax_poor.set_xticks(x_indices, PATCH_LABELS)
ax_poor.set_xlabel('Patch Type (Initial Yield)')
ax_poor.set_title('Milk Task: Poor Environment')
ax_poor.legend(loc='best')
ax_poor.grid(True, linestyle=':', alpha=0.7)

# Adjust y-axis limits
all_values = []
for arr in simulated_risk_ET_rich.values():
    all_values.extend(arr)
for arr in simulated_risk_ET_poor.values():
    all_values.extend(arr)
all_values.extend(T_OPT_RICH_ENV)
all_values.extend(T_OPT_POOR_ENV)
all_values.extend(T_real_dev_rich)
all_values.extend(T_real_dev_poor)

min_val = max(0, np.min(all_values) - 5)
max_val = np.max(all_values) + 10
axes[0].set_ylim(bottom=min_val, top=max_val)

fig.suptitle(
    f"Milk Task: MVT, Real Deviation, & Simulated Risk Profiles\n"
    f"(c_0={C0_BASELINE}, K={K_RISK_SENSITIVITY}, β_rich={BETA_MILK_RICH}, β_poor={BETA_MILK_POOR})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Table
import numpy as np
import pandas as pd

# Define patch mid-yield index
patch_mid = 1  # 0=low, 1=mid, 2=high

# MVT optimum and “real” suboptimal times for mid-yield
T_opt_rich = np.array([5.28,  9.62, 12.89])
T_opt_poor = np.array([7.46, 11.80, 15.07])
bias_rich  = 5.9
bias_poor  = 7.1

T_real_rich = T_opt_rich + bias_rich
T_real_poor = T_opt_poor + bias_poor

# Analytic E[T] for mid-yield
E_T_ana_rich = {'Risk Averse': 28.15, 'Risk Seeking': 13.98}
E_T_ana_poor = {'Risk Averse': 29.24, 'Risk Seeking': 15.07}

# Build the table rows
rows = []
for env, Topt, Treal, E_T_ana in [
    ('Rich', T_opt_rich[patch_mid], T_real_rich[patch_mid], E_T_ana_rich),
    ('Poor', T_opt_poor[patch_mid], T_real_poor[patch_mid], E_T_ana_poor),
]:
    for profile in ['Risk Averse', 'Risk Seeking']:
        E = E_T_ana[profile]
        rows.append({
            'Environment':    env,
            'Profile':        profile,
            'T* (opt) (s)':   Topt,
            'T* (real) (s)':  Treal,
            'E[T]_ana (s)':   E,
            'Δ_opt (s)':      E - Topt,
            'Δ_real (s)':     E - Treal,
        })

df = pd.DataFrame(rows)

# Print plain-text table
print("\nMid-Yield Distance Table (Signed Differences)\n")
print(df.to_string(index=False, float_format="%.2f"))








