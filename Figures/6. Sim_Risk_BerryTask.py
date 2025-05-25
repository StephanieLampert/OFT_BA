import numpy as np
import matplotlib.pyplot as plt

# --- Core Model & Task Parameters ---
# Berry Task (Contreras-Huerta et al. parameters)
PATCH_INITIAL_YIELDS = np.array([34.5, 57.5])  # Low, High
PATCH_LABELS = ['Low (34.5)', 'High (57.5)']
LAMBDA_DECAY = 0.11  # Decay rate for Berry task

# MVT Optimal Leaving Times for Berry Task
MVT_OPTIMAL_RICH_VALS = np.array([3.39, 8.04])  # T_opt for Low, High in Rich
MVT_OPTIMAL_POOR_VALS = np.array([5.30, 9.94])  # T_opt for Low, High in Poor

# Agent Parameters
C0_BASELINE = -2.0
K_RISK_SENSITIVITY = 2.0
BETA_REWARD_SENSITIVITY = 0.5

# Simulation settings
MAX_TIME_STEPS = 100
TIME_STEPS = np.arange(1, MAX_TIME_STEPS + 1)

# Alpha profiles for this specific plot
ALPHA_PROFILES_FOR_PLOT = {
    'Risk Averse (α=0.8)': 0.8,
    'Risk Seeking (α=1.2)': 1.2,
}


# --- Helper Functions ---
def calculate_reward_rate_at_time(s_initial, t, lam):
    return s_initial * np.exp(-lam * t)


def get_leaving_probabilities_and_survival(c_eff, beta, s_initial, time_steps, lam_decay_task):
    current_rewards = calculate_reward_rate_at_time(s_initial, time_steps, lam=lam_decay_task)
    logits = c_eff + beta * current_rewards
    prob_leave_at_n = 1 / (1 + np.exp(logits))
    prob_stay_sequence = 1.0 - prob_leave_at_n
    prob_not_left_before_t = np.concatenate(([1.0], np.cumprod(prob_stay_sequence)))[:-1]
    return prob_leave_at_n, prob_not_left_before_t


def calculate_expected_leaving_time(prob_leave_at_n, prob_not_left_before_t, time_steps):
    prob_leave_exactly_at_n = prob_leave_at_n * prob_not_left_before_t
    expected_T = np.sum(time_steps * prob_leave_exactly_at_n)
    return expected_T


# --- Calculate E[T] for the specified Alpha Profiles ---
simulated_ET_results = {}

print(f"Calculating E[T] for specified Alpha profiles (Berry Task, K={K_RISK_SENSITIVITY}, Corrected MVT)...\n")
for profile_name, alpha_val in ALPHA_PROFILES_FOR_PLOT.items():
    print(f"Profile: {profile_name} (alpha = {alpha_val})")
    c_eff = C0_BASELINE + K_RISK_SENSITIVITY * (1 - alpha_val)
    print(f"  Calculated c_eff = {c_eff:.4f}")

    profile_ET_values = []
    for s_initial in PATCH_INITIAL_YIELDS:
        prob_leave_at_n, prob_not_left_before_t = get_leaving_probabilities_and_survival(
            c_eff, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS, LAMBDA_DECAY
        )
        ET = calculate_expected_leaving_time(
            prob_leave_at_n, prob_not_left_before_t, TIME_STEPS
        )
        profile_ET_values.append(ET)
        patch_idx = list(PATCH_INITIAL_YIELDS).index(s_initial)
        print(f"  Patch: {PATCH_LABELS[patch_idx]:<15} | E[T] = {ET:>6.2f}s")

    simulated_ET_results[profile_name] = np.array(profile_ET_values)
    print("-" * 50)

# --- Plotting: Deviation from MVT for Berry Task ---
plt.figure(figsize=(10, 7))
x_indices = np.arange(len(PATCH_LABELS))

# MVT Optimal Lines
plt.plot(x_indices, MVT_OPTIMAL_RICH_VALS, 'o--', color='gold', lw=2, markersize=8, label='MVT Optimal (Rich Env)')
plt.plot(x_indices, MVT_OPTIMAL_POOR_VALS, 's--', color='limegreen', lw=2, markersize=8, label='MVT Optimal (Poor Env)')

# Simulated Agent E[T] Lines
colors = {'Risk Averse (α=0.8)': '#1f77b4', 'Risk Seeking (α=1.2)': '#ff7f0e'}
markers = {'Risk Averse (α=0.8)': 'P', 'Risk Seeking (α=1.2)': 'X'}

for profile_name, ET_values in simulated_ET_results.items():
    plt.plot(x_indices, ET_values, marker=markers[profile_name], linestyle='-', color=colors[profile_name],
             lw=2.5, markersize=9, label=f'{profile_name}')

plt.xticks(x_indices, PATCH_LABELS)
plt.xlabel('Patch Type (Initial Yield)')
plt.ylabel('Time in Patch (seconds)')
plt.title(
    f'Berry Task: Time in Patch for Different Risk Profiles\n($c_0={C0_BASELINE}, K={K_RISK_SENSITIVITY}, \\beta={BETA_REWARD_SENSITIVITY}$)')
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()




