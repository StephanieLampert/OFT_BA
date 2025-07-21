import numpy as np
import matplotlib.pyplot as plt

# --- Core Model & Task Parameters (Berry Task) ---
PATCH_INITIAL_YIELDS = np.array([34.5, 57.5])    # Low, High
PATCH_LABELS = ['Low (34.5)', 'High (57.5)']
LAMBDA_DECAY_TASK = 0.11                         # Decay rate for Berry task

# --- MVT Optimal Leaving Times for Berry Task (from Contreras‐Huerta) ---
T_OPT_RICH_ENV = np.array([3.39, 8.04])          # T* for Low, High in Rich
T_OPT_POOR_ENV = np.array([5.30, 9.94])          # T* for Low, High in Poor

# --- “Real Foraging Deviation” Biases (Contreras‐Huerta SI) ---
BIAS_RICH_ENV = 5.30
BIAS_POOR_ENV = 5.49

# --- Environment‐specific β (Berry Task) from BRR calculations ---
#   BRR_Rich ≈ 23.7513 → β_Rich ≈ 0.4719
#   BRR_Poor ≈ 19.2604 → β_Poor ≈ 0.5241
BETA_BERRY_RICH = 0.4719
BETA_BERRY_POOR = 0.5241

# --- Risk‐Profile Parameters (same as Milk Task) ---
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
    # Avoid zero rewards for log/exp stability
    current_rewards = np.maximum(current_rewards, 1e-9)
    logits = c_eff + beta * current_rewards
    # Clip logits to prevent overflow
    logits = np.clip(logits, -500, 500)
    prob_leave_at_n = 1.0 / (1.0 + np.exp(logits))
    prob_stay_sequence = 1.0 - prob_leave_at_n
    prob_stay_sequence = np.minimum(prob_stay_sequence, 1.0 - 1e-12)
    # Probability of still being in patch at time t
    prob_not_left_before_t = np.concatenate(([1.0], np.cumprod(prob_stay_sequence)))[:-1]
    return prob_leave_at_n, prob_not_left_before_t

def calculate_expected_leaving_time(prob_leave_at_n, prob_not_left_before_t, time_steps_arr):
    prob_leave_exactly_at_n = prob_leave_at_n * prob_not_left_before_t
    expected_T = np.sum(time_steps_arr * prob_leave_exactly_at_n)
    # Check for possible truncation
    if expected_T > MAX_TIME_STEPS * 0.98:
        print(f"Warning: E[T] ({expected_T:.2f}s) is close to MAX_TIME_STEPS ({MAX_TIME_STEPS}s).")
    return expected_T

# --- 1. “Real Foraging Deviation” Times (MVT + Bias) ---
T_real_dev_rich = T_OPT_RICH_ENV + BIAS_RICH_ENV
T_real_dev_poor = T_OPT_POOR_ENV + BIAS_POOR_ENV

# --- 2. Calculate E[T] for Simulated Risk Profiles in Rich & Poor ---
simulated_risk_ET_rich = {}
simulated_risk_ET_poor = {}

print(f"Simulating Berry Task with: C0={C0_BASELINE}, K={K_RISK_SENSITIVITY}\n")
for profile_name, alpha_val in ALPHA_PROFILES_RISK_SIM.items():
    # Compute effective bias c_eff for this alpha
    c_eff = C0_BASELINE + K_RISK_SENSITIVITY * (1 - alpha_val)
    et_rich_values = []
    et_poor_values = []
    print(f"Profile: {profile_name} (α={alpha_val}, c_eff={c_eff:.2f})")
    for s_initial in PATCH_INITIAL_YIELDS:
        # Rich environment calculation
        prob_leave_r, prob_not_left_r = get_leaving_probabilities_and_survival(
            c_eff, BETA_BERRY_RICH, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK
        )
        ET_rich = calculate_expected_leaving_time(prob_leave_r, prob_not_left_r, TIME_STEPS_ET)
        et_rich_values.append(ET_rich)

        # Poor environment calculation
        prob_leave_p, prob_not_left_p = get_leaving_probabilities_and_survival(
            c_eff, BETA_BERRY_POOR, s_initial, TIME_STEPS_ET, LAMBDA_DECAY_TASK
        )
        ET_poor = calculate_expected_leaving_time(prob_leave_p, prob_not_left_p, TIME_STEPS_ET)
        et_poor_values.append(ET_poor)

        patch_idx = list(PATCH_INITIAL_YIELDS).index(s_initial)
        print(f"  Patch: {PATCH_LABELS[patch_idx]:<15} | E[T]_rich = {ET_rich:>6.2f}s | E[T]_poor = {ET_poor:>6.2f}s")

    simulated_risk_ET_rich[profile_name] = np.array(et_rich_values)
    simulated_risk_ET_poor[profile_name] = np.array(et_poor_values)
    print("-" * 60)

# --- 3. Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
x_indices = np.arange(len(PATCH_LABELS))

plot_styles = {
    'MVT Optimal':         {'color': 'forestgreen',   'linestyle': '-',  'marker': 'o', 'linewidth': 2,   'markersize': 7},
    'Real Foraging':       {'color': 'purple',        'linestyle': '-',  'marker': 'o', 'linewidth': 2,   'markersize': 7},
    'Risk Averse (α=0.5)': {'color': 'blue',          'linestyle': ':',  'marker': 'o', 'linewidth': 2.5, 'markersize': 9},
    'Risk Seeking (α=1.5)':{'color': 'darkorange',    'linestyle': ':',  'marker': 'o', 'linewidth': 2.5, 'markersize': 9},
}

# Plot for Rich Environment
ax_rich = axes[0]
ax_rich.plot(x_indices, T_OPT_RICH_ENV,       **plot_styles['MVT Optimal'],      label='MVT Optimal (Rich)')
ax_rich.plot(x_indices, T_real_dev_rich,      **plot_styles['Real Foraging'],    label='Real Foraging (Rich)')
ax_rich.plot(
    x_indices,
    simulated_risk_ET_rich['Risk Averse (α=0.5)'],
    **plot_styles['Risk Averse (α=0.5)'],
    label='Sim: Risk Averse (α=0.5)'
)
ax_rich.plot(
    x_indices,
    simulated_risk_ET_rich['Risk Seeking (α=1.5)'],
    **plot_styles['Risk Seeking (α=1.5)'],
    label='Sim: Risk Seeking (α=1.5)'
)

ax_rich.set_xticks(x_indices, PATCH_LABELS)
ax_rich.set_xlabel('Patch Type (Initial Yield)')
ax_rich.set_ylabel('Time in Patch (seconds)')
ax_rich.set_title('Berry Task: Rich Environment')
ax_rich.legend(loc='best')
ax_rich.grid(True, linestyle=':', alpha=0.7)

# Plot for Poor Environment
ax_poor = axes[1]
ax_poor.plot(x_indices, T_OPT_POOR_ENV,       **plot_styles['MVT Optimal'],      label='MVT Optimal (Poor)')
ax_poor.plot(x_indices, T_real_dev_poor,      **plot_styles['Real Foraging'],    label='Real Foraging (Poor)')
ax_poor.plot(
    x_indices,
    simulated_risk_ET_poor['Risk Averse (α=0.5)'],
    **plot_styles['Risk Averse (α=0.5)'],
    label='Sim: Risk Averse (α=0.5)'
)
ax_poor.plot(
    x_indices,
    simulated_risk_ET_poor['Risk Seeking (α=1.5)'],
    **plot_styles['Risk Seeking (α=1.5)'],
    label='Sim: Risk Seeking (α=1.5)'
)

ax_poor.set_xticks(x_indices, PATCH_LABELS)
ax_poor.set_xlabel('Patch Type (Initial Yield)')
ax_poor.set_title('Berry Task: Poor Environment')
ax_poor.legend(loc='best')
ax_poor.grid(True, linestyle=':', alpha=0.7)

# Adjust y-axis limits based on all plotted values
all_et_values = []
for arr in simulated_risk_ET_rich.values():
    all_et_values.extend(arr)
for arr in simulated_risk_ET_poor.values():
    all_et_values.extend(arr)
all_et_values.extend(T_OPT_RICH_ENV)
all_et_values.extend(T_OPT_POOR_ENV)
all_et_values.extend(T_real_dev_rich)
all_et_values.extend(T_real_dev_poor)

if all_et_values:
    valid_vals = [v for v in all_et_values if np.isfinite(v)]
    y_min = max(0, np.min(valid_vals) - 5)
    y_max = np.max(valid_vals) + 10
    axes[0].set_ylim(bottom=y_min, top=y_max)

fig.suptitle(
    f"Berry Task: MVT, Real Deviation, & Simulated Risk Profiles\n"
    f"(c_0={C0_BASELINE}, K={K_RISK_SENSITIVITY}, β_rich={BETA_BERRY_RICH}, β_poor={BETA_BERRY_POOR})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#Table
import numpy as np
import pandas as pd

# --- MVT Optima & Real Deviations for Berry Task (high yield only) ---
T_opt_rich_high = 8.04   # rich, high yield
T_opt_poor_high = 9.94   # poor, high yield
bias_rich = 5.30
bias_poor = 5.49

T_real_rich_high = T_opt_rich_high + bias_rich
T_real_poor_high = T_opt_poor_high + bias_poor

# --- Analytic expected leave times (high yield only) ---
E_T_ana_rich_high = {
    'Risk Averse': 22.20,
    'Risk Seeking': 11.88
}
E_T_ana_poor_high = {
    'Risk Averse': 23.15,
    'Risk Seeking': 12.83
}

# Build rows for just high yield
rows = []
for env in ['Rich', 'Poor']:
    Topt = T_opt_rich_high if env=='Rich' else T_opt_poor_high
    Treal = T_real_rich_high if env=='Rich' else T_real_poor_high
    Edict = E_T_ana_rich_high if env=='Rich' else E_T_ana_poor_high

    for profile, Eana in Edict.items():
        rows.append({
            'Environment': env,
            'Profile':     profile,
            'T* (opt)':    Topt,
            'T* (real)':   Treal,
            'E[T]_ana':    Eana,
            'Δ to opt':    Eana - Topt,
            'Δ to real':   Eana - Treal
        })

df = pd.DataFrame(rows)

# Print as aligned table
print(df.to_string(index=False, float_format="%.2f"))










