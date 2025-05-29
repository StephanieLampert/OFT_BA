import numpy as np
import matplotlib.pyplot as plt

# --- Task Parameters (Milk Task - Le Heron et al.) ---
PATCH_INITIAL_YIELDS = np.array([32.5, 45.0, 57.5])
PATCH_LABELS_PRINT = ['Low Yield', 'Mid Yield', 'High Yield']
LAMBDA_DECAY_TASK = 0.075

# --- Agent Simulation Parameters (Fixed $c_0$, $\beta$) ---
C0_BASELINE = -4.0
BETA_REWARD_SENSITIVITY = 0.5
ALPHA_VALUES = {
    'Risk Averse': 0.5,
    'Risk Seeking': 1.5,
}

# --- K Values to Compare ---
K_VALUES_TO_COMPARE = [2.0, 4.0, 6.0]
K_LABELS_X_AXIS = [f'K={K_VALUES_TO_COMPARE[0]}', f'K={K_VALUES_TO_COMPARE[1]}', f'K={K_VALUES_TO_COMPARE[2]}']

# Simulation settings
MAX_TIME_STEPS = 100
TIME_STEPS_ET = np.arange(1, MAX_TIME_STEPS + 1)


# --- Helper Functions (Keep these as they are) ---
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
        print(
            f"Warning: E[T] ({expected_T:.2f}s) is close to MAX_TIME_STEPS ({MAX_TIME_STEPS}s). Result might be truncated.")
    return expected_T


# --- Calculate E[T] and Average Discrepancies ---
average_discrepancy_results = {}

print(f"Calculating E[T] and Average Discrepancies for K values (Milk Task)\n")
print(f"Agent parameters: C0={C0_BASELINE}, Beta={BETA_REWARD_SENSITIVITY}")
print(f"Alpha Averse={ALPHA_VALUES['Risk Averse']}, Alpha Seeking={ALPHA_VALUES['Risk Seeking']}\n")

for K_val_current, k_label_x_axis in zip(K_VALUES_TO_COMPARE, K_LABELS_X_AXIS):
    print(f"Simulating for {k_label_x_axis}")
    ET_averse_current_K = []
    ET_seeking_current_K = []

    alpha_averse = ALPHA_VALUES['Risk Averse']
    c_eff_averse = C0_BASELINE + K_val_current * (1 - alpha_averse)
    for s_initial in PATCH_INITIAL_YIELDS:
        prob_leave_n, prob_not_left_before = get_leaving_probabilities_and_survival(
            c_eff_averse, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK)
        ET_averse = calculate_expected_leaving_time(prob_leave_n, prob_not_left_before, TIME_STEPS_ET)
        ET_averse_current_K.append(ET_averse)

    alpha_seeking = ALPHA_VALUES['Risk Seeking']
    c_eff_seeking = C0_BASELINE + K_val_current * (1 - alpha_seeking)
    for s_initial in PATCH_INITIAL_YIELDS:
        prob_leave_n, prob_not_left_before = get_leaving_probabilities_and_survival(
            c_eff_seeking, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK)
        ET_seeking = calculate_expected_leaving_time(prob_leave_n, prob_not_left_before, TIME_STEPS_ET)
        ET_seeking_current_K.append(ET_seeking)

    discrepancies_per_patch = np.array(ET_averse_current_K) - np.array(ET_seeking_current_K)
    average_discrepancy = np.mean(discrepancies_per_patch)
    average_discrepancy_results[k_label_x_axis] = average_discrepancy

    print(f"  Discrepancies for {k_label_x_axis}:")
    for i, plabel in enumerate(PATCH_LABELS_PRINT):
        print(f"    {plabel}: {discrepancies_per_patch[i]:.2f}s")
    print(f"  Average Discrepancy for {k_label_x_axis}: {average_discrepancy:.2f}s\n")
    print("-" * 50)

# --- Plotting Bar Chart of Average Discrepancies ---
fig, ax = plt.subplots(figsize=(10, 7))

k_categories = list(average_discrepancy_results.keys())
avg_discrepancy_values = list(average_discrepancy_results.values())

# --- MODIFIED bar_colors for a "less pastel" / stronger palette ---
bar_colors = ['royalblue', 'orangered', 'forestgreen']  # K=2.0, K=4.0, K=6.0

ax.bar(k_categories, avg_discrepancy_values, color=bar_colors)

ax.set_ylabel('Average Discrepancy in $E[T]$ ($E[T]_{Averse} - E[T]_{Seeking}$) (s)')
ax.set_xlabel('K Value (Risk Sensitivity Scaler)')
ax.set_title(f' Milk Task: Effect of K on Risk Profiles\n'
             f'($c_0={C0_BASELINE}, \\beta={BETA_REWARD_SENSITIVITY}, \\alpha_{{Averse}}={ALPHA_VALUES["Risk Averse"]}, \\alpha_{{Seeking}}={ALPHA_VALUES["Risk Seeking"]}$)',
             fontsize=11)
ax.grid(True, linestyle=':', axis='y', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)

# Add text labels on top of bars (without "s")
for i, v in enumerate(avg_discrepancy_values):
    offset_factor = 0.02 * np.max(np.abs(avg_discrepancy_values)) if np.max(
        np.abs(avg_discrepancy_values)) != 0 else 0.5
    offset = offset_factor if v >= 0 else -offset_factor * 2
    ax.text(i, v + offset, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

fig.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- Task Parameters (Berry Task - Contreras-Huerta et al. parameters) ---
PATCH_INITIAL_YIELDS = np.array([34.5, 57.5])  # Low, High
PATCH_LABELS_PRINT = ['Low Yield (34.5)', 'High Yield (57.5)'] # Updated for Berry Task
LAMBDA_DECAY_TASK = 0.11  # Decay rate for Berry task

# --- Agent Simulation Parameters (Fixed $c_0$, $\beta$) ---
C0_BASELINE = -4.0
BETA_REWARD_SENSITIVITY = 0.5
ALPHA_VALUES = {
    'Risk Averse': 0.5,
    'Risk Seeking': 1.5,
}

# --- K Values to Compare ---
K_VALUES_TO_COMPARE = [2.0, 4.0, 6.0]
K_LABELS_X_AXIS = [f'K={K_VALUES_TO_COMPARE[0]}', f'K={K_VALUES_TO_COMPARE[1]}', f'K={K_VALUES_TO_COMPARE[2]}']

# Simulation settings
MAX_TIME_STEPS = 100
TIME_STEPS_ET = np.arange(1, MAX_TIME_STEPS + 1)


# --- Helper Functions (Keep these as they are) ---
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
        print(
            f"Warning: E[T] ({expected_T:.2f}s) is close to MAX_TIME_STEPS ({MAX_TIME_STEPS}s). Result might be truncated.")
    return expected_T


# --- Calculate E[T] and Average Discrepancies ---
average_discrepancy_results = {}

print(f"Calculating E[T] and Average Discrepancies for K values (Berry Task)\n") # Updated Task Name
print(f"Agent parameters: C0={C0_BASELINE}, Beta={BETA_REWARD_SENSITIVITY}")
print(f"Alpha Averse={ALPHA_VALUES['Risk Averse']}, Alpha Seeking={ALPHA_VALUES['Risk Seeking']}\n")

for K_val_current, k_label_x_axis in zip(K_VALUES_TO_COMPARE, K_LABELS_X_AXIS):
    print(f"Simulating for {k_label_x_axis}")
    ET_averse_current_K = []
    ET_seeking_current_K = []

    alpha_averse = ALPHA_VALUES['Risk Averse']
    c_eff_averse = C0_BASELINE + K_val_current * (1 - alpha_averse)
    for s_initial in PATCH_INITIAL_YIELDS: # Will iterate over 2 patch yields for Berry Task
        prob_leave_n, prob_not_left_before = get_leaving_probabilities_and_survival(
            c_eff_averse, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK)
        ET_averse = calculate_expected_leaving_time(prob_leave_n, prob_not_left_before, TIME_STEPS_ET)
        ET_averse_current_K.append(ET_averse)

    alpha_seeking = ALPHA_VALUES['Risk Seeking']
    c_eff_seeking = C0_BASELINE + K_val_current * (1 - alpha_seeking)
    for s_initial in PATCH_INITIAL_YIELDS: # Will iterate over 2 patch yields for Berry Task
        prob_leave_n, prob_not_left_before = get_leaving_probabilities_and_survival(
            c_eff_seeking, BETA_REWARD_SENSITIVITY, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK)
        ET_seeking = calculate_expected_leaving_time(prob_leave_n, prob_not_left_before, TIME_STEPS_ET)
        ET_seeking_current_K.append(ET_seeking)

    discrepancies_per_patch = np.array(ET_averse_current_K) - np.array(ET_seeking_current_K)
    average_discrepancy = np.mean(discrepancies_per_patch) # Average will be over 2 values now
    average_discrepancy_results[k_label_x_axis] = average_discrepancy

    print(f"  Discrepancies for {k_label_x_axis}:")
    for i, plabel in enumerate(PATCH_LABELS_PRINT): # Uses updated PATCH_LABELS_PRINT
        print(f"    {plabel}: {discrepancies_per_patch[i]:.2f}s")
    print(f"  Average Discrepancy for {k_label_x_axis}: {average_discrepancy:.2f}s\n")
    print("-" * 50)

# --- Plotting Bar Chart of Average Discrepancies ---
fig, ax = plt.subplots(figsize=(10, 7))

k_categories = list(average_discrepancy_results.keys())
avg_discrepancy_values = list(average_discrepancy_results.values())

bar_colors = ['royalblue', 'orangered', 'forestgreen']  # K=2.0, K=4.0, K=6.0

ax.bar(k_categories, avg_discrepancy_values, color=bar_colors)

ax.set_ylabel('Average Discrepancy in $E[T]$ ($E[T]_{Averse} - E[T]_{Seeking}$) (s)')
ax.set_xlabel('K Value (Risk Sensitivity Scaler)')
ax.set_title(f'Berry Task: Effect of K on Risk Profiles\n' # Updated Task Name
             f'($c_0={C0_BASELINE}, \\beta={BETA_REWARD_SENSITIVITY}, \\alpha_{{Averse}}={ALPHA_VALUES["Risk Averse"]}, \\alpha_{{Seeking}}={ALPHA_VALUES["Risk Seeking"]}$)',
             fontsize=11)
ax.grid(True, linestyle=':', axis='y', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)

# Add text labels on top of bars (without "s")
for i, v in enumerate(avg_discrepancy_values):
    offset_factor = 0.02 * np.max(np.abs(avg_discrepancy_values)) if np.max(
        np.abs(avg_discrepancy_values)) != 0 else 0.5
    offset = offset_factor if v >= 0 else -offset_factor * 2
    ax.text(i, v + offset, f"{v:.2f}", ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')

fig.tight_layout()
plt.show()