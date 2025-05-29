import numpy as np
import matplotlib.pyplot as plt

# --- Core Model & Task Parameters ---
# Berry Task (Contreras-Huerta et al. parameters)
PATCH_INITIAL_YIELDS = np.array([34.5, 57.5])  # Low, High
PATCH_LABELS = ['Low (34.5)', 'High (57.5)']
LAMBDA_DECAY_TASK = 0.11  # Decay rate for Berry task

# MVT Optimal Leaving Times for Berry Task (from your confirmed values)
T_OPT_RICH_ENV = np.array([3.39, 8.04])  # T_opt for Low, High in Rich
T_OPT_POOR_ENV = np.array([5.30, 9.94])  # T_opt for Low, High in Poor

# --- Parameters for "Real Foraging Deviation" (from Contreras-Huerta SI) ---
BIAS_RICH_ENV = 5.30
BIAS_POOR_ENV = 5.49

# --- Parameters for "Simulation including Risk Preference" ---
# Kept consistent with the last Milk Task version for the simulated agent's properties
C0_BASELINE = -4.0
K_RISK_SENSITIVITY = 6.0
BETA_REWARD_SENSITIVITY = 0.5

# Alpha profiles
ALPHA_PROFILES_RISK_SIM = {
    'Risk Averse (α=0.5)': 0.5,
    'Risk Seeking (α=1.5)': 1.5,
}

# Simulation settings
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
    if sum_probs < 0.99 and MAX_TIME_STEPS > 10 :
        pass
    expected_T = np.sum(time_steps_arr * prob_leave_exactly_at_n)
    if expected_T > MAX_TIME_STEPS * 0.98 and MAX_TIME_STEPS > 10:
        print(f"Warning: E[T] ({expected_T:.2f}s) is close to MAX_TIME_STEPS ({MAX_TIME_STEPS}s). Result might be truncated for some profiles.")
    return expected_T

# --- 1. "Real Foraging Deviation" Times (MVT + Bias) ---
T_real_dev_rich = T_OPT_RICH_ENV + BIAS_RICH_ENV
T_real_dev_poor = T_OPT_POOR_ENV + BIAS_POOR_ENV

# --- 2. Calculate E[T] for Simulated Risk Profiles ---
simulated_risk_ET_results = {}
print(f"Simulating Berry Task with: C0={C0_BASELINE}, K={K_RISK_SENSITIVITY}, Beta={BETA_REWARD_SENSITIVITY}\n")
for profile_name, alpha_val in ALPHA_PROFILES_RISK_SIM.items():
    c_eff = C0_BASELINE + K_RISK_SENSITIVITY * (1 - alpha_val)
    profile_ET_values = []
    print(f"Profile: {profile_name} (alpha={alpha_val}, c_eff={c_eff:.2f})")
    for s_initial in PATCH_INITIAL_YIELDS:
        prob_leave_n, prob_not_left_before = get_leaving_probabilities_and_survival(
            c_eff, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK
        )
        ET = calculate_expected_leaving_time(
            prob_leave_n, prob_not_left_before, TIME_STEPS_ET
        )
        profile_ET_values.append(ET)
        patch_idx = list(PATCH_INITIAL_YIELDS).index(s_initial)
        print(f"  Patch: {PATCH_LABELS[patch_idx]:<15} | E[T] = {ET:>6.2f}s")
    simulated_risk_ET_results[profile_name] = np.array(profile_ET_values)
    print("-" * 30)

# --- 3. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
x_indices = np.arange(len(PATCH_LABELS))

# Plotting styles from last Milk Task version
plot_elements = {
    'MVT Optimal': {'color': 'forestgreen', 'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'markersize': 7},
    'Real Foraging': {'color': 'purple', 'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'markersize': 7},
    'Risk Averse (α=0.5)': {'color': 'blue', 'linestyle': ':', 'marker': 'o', 'linewidth': 2.5, 'markersize': 9},
    'Risk Seeking (α=1.5)': {'color': 'darkorange', 'linestyle': ':', 'marker': 'o', 'linewidth': 2.5, 'markersize': 9}
}

# Plot for Rich Environment
ax_rich = axes[0]
ax_rich.plot(x_indices, T_OPT_RICH_ENV, **plot_elements['MVT Optimal'], label='MVT Optimal (Rich)')
ax_rich.plot(x_indices, T_real_dev_rich, **plot_elements['Real Foraging'], label='Real Foraging (Rich)')
ax_rich.plot(x_indices, simulated_risk_ET_results['Risk Averse (α=0.5)'], **plot_elements['Risk Averse (α=0.5)'], label='Sim: Risk Averse (α=0.5)')
ax_rich.plot(x_indices, simulated_risk_ET_results['Risk Seeking (α=1.5)'], **plot_elements['Risk Seeking (α=1.5)'], label='Sim: Risk Seeking (α=1.5)')

ax_rich.set_xticks(x_indices, PATCH_LABELS)
ax_rich.set_xlabel('Patch Type (Initial Yield)')
ax_rich.set_ylabel('Time in Patch (seconds)')
ax_rich.set_title(f'Berry Task: Rich Environment')
ax_rich.legend(loc='best')
ax_rich.grid(True, linestyle=':', alpha=0.7)

# Plot for Poor Environment
ax_poor = axes[1]
ax_poor.plot(x_indices, T_OPT_POOR_ENV, **plot_elements['MVT Optimal'], label='MVT Optimal (Poor)')
ax_poor.plot(x_indices, T_real_dev_poor, **plot_elements['Real Foraging'], label='Real Foraging (Poor)')
ax_poor.plot(x_indices, simulated_risk_ET_results['Risk Averse (α=0.5)'], **plot_elements['Risk Averse (α=0.5)'], label='Sim: Risk Averse (α=0.5)')
ax_poor.plot(x_indices, simulated_risk_ET_results['Risk Seeking (α=1.5)'], **plot_elements['Risk Seeking (α=1.5)'], label='Sim: Risk Seeking (α=1.5)')

ax_poor.set_xticks(x_indices, PATCH_LABELS)
ax_poor.set_xlabel('Patch Type (Initial Yield)')
ax_poor.set_title(f'Berry Task: Poor Environment')
ax_poor.legend(loc='best')
ax_poor.grid(True, linestyle=':', alpha=0.7)

# Adjust y-axis
all_et_values_plot = []
for key_profile in simulated_risk_ET_results:
    all_et_values_plot.extend(simulated_risk_ET_results[key_profile])
all_et_values_plot.extend(T_OPT_RICH_ENV)
all_et_values_plot.extend(T_OPT_POOR_ENV)
all_et_values_plot.extend(T_real_dev_rich)
all_et_values_plot.extend(T_real_dev_poor)

min_val_plot = 0
max_val_plot = MAX_TIME_STEPS
if all_et_values_plot:
    valid_values = [v for v in all_et_values_plot if np.isfinite(v)]
    if valid_values:
        min_val_plot = np.min(valid_values)
        max_val_plot = np.max(valid_values)

axes[0].set_ylim(bottom=max(0, min_val_plot - 5), top=max_val_plot + 10)

fig.suptitle(f'Berry Task: MVT, Real Deviation, & Simulated Risk Profiles\n($c_0={C0_BASELINE}, K={K_RISK_SENSITIVITY}, \\beta={BETA_REWARD_SENSITIVITY}$)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




