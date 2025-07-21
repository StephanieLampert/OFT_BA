import numpy as np
import matplotlib.pyplot as plt

# --- Agent Simulation Parameters (Fixed for both tasks) ---
C0_BASELINE = -4.0
BETA_REWARD_SENSITIVITY = 0.5
ALPHA_VALUES = {
    'Risk Averse': 0.5,
    'Risk Seeking': 1.5,
}
K_VALUES_TO_COMPARE = [2.0, 4.0, 6.0]
K_LABELS_X_AXIS = [f'K={K}' for K in K_VALUES_TO_COMPARE]

# --- Task-Specific Parameters ---
task_params = {
    'Milk Task': {
        'initial_yields': np.array([32.5, 45.0, 57.5]),
        'lambda_decay': 0.075
    },
    'Berry Task': {
        'initial_yields': np.array([34.5, 57.5]),
        'lambda_decay': 0.11
    }
}

# Simulation settings
MAX_TIME_STEPS = 100
TIME_STEPS_ET = np.arange(1, MAX_TIME_STEPS + 1)

# Helper functions
def calculate_expected_leaving_time(c_eff, beta, s_initial, lam):
    rewards = s_initial * np.exp(-lam * TIME_STEPS_ET)
    logits = np.clip(c_eff + beta * rewards, -500, 500)
    p_leave = 1 / (1 + np.exp(logits))
    p_stay = 1 - p_leave
    survival = np.concatenate(([1.0], np.cumprod(p_stay)))[:-1]
    return np.sum(TIME_STEPS_ET * (p_leave * survival))

# Calculate discrepancies
all_tasks_discrepancy_results = {}
for task_name, params in task_params.items():
    avg_discrepancy_results_per_k = {}
    for K_val_current, k_label in zip(K_VALUES_TO_COMPARE, K_LABELS_X_AXIS):
        et_averse = [
            calculate_expected_leaving_time(C0_BASELINE + K_val_current*(1-ALPHA_VALUES['Risk Averse']),
                                            BETA_REWARD_SENSITIVITY, s, params['lambda_decay'])
            for s in params['initial_yields']
        ]
        et_seeking = [
            calculate_expected_leaving_time(C0_BASELINE + K_val_current*(1-ALPHA_VALUES['Risk Seeking']),
                                            BETA_REWARD_SENSITIVITY, s, params['lambda_decay'])
            for s in params['initial_yields']
        ]
        avg_discrepancy_results_per_k[k_label] = np.mean(np.array(et_averse) - np.array(et_seeking))
    all_tasks_discrepancy_results[task_name] = avg_discrepancy_results_per_k

# Plotting
fig, (ax_milk, ax_berry) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
bar_colors = ['royalblue', 'orangered', 'forestgreen']

# Milk Task
milk_results = all_tasks_discrepancy_results['Milk Task']
bars = ax_milk.bar(K_LABELS_X_AXIS, list(milk_results.values()), color=bar_colors)
for bar in bars:
    x, height = bar.get_x(), bar.get_height()
    ax_milk.text(x + bar.get_width()/2, height/2, f"{height:.2f}", ha='center', va='center', color='white', fontweight='bold')
ax_milk.set_title('Milk Task')
ax_milk.set_ylabel('Average Discrepancy in $E[leave]$ ($E[leave]_{Averse} - E[leave]_{Seeking}$) (s)')
ax_milk.set_xlabel('K Value')
ax_milk.grid(True, axis='y', linestyle=':', alpha=0.7)

# Berry Task
berry_results = all_tasks_discrepancy_results['Berry Task']
bars = ax_berry.bar(K_LABELS_X_AXIS, list(berry_results.values()), color=bar_colors)
for bar in bars:
    x, height = bar.get_x(), bar.get_height()
    ax_berry.text(x + bar.get_width()/2, height/2, f"{height:.2f}", ha='center', va='center', color='white', fontweight='bold')
ax_berry.set_title('Berry Task')
ax_berry.set_xlabel('K Value')
ax_berry.grid(True, axis='y', linestyle=':', alpha=0.7)

fig.suptitle(f'Effect of Variation in K\n($c_0={C0_BASELINE}, \\beta={BETA_REWARD_SENSITIVITY}$)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
