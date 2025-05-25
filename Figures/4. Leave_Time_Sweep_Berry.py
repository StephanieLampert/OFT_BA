import numpy as np
import matplotlib.pyplot as plt

# Berry task parameters
lam     = 0.11
patches = [34.5, 57.5]
labels  = ['Low (34.5)', 'High (57.5)']
colors  = ['C0', 'C2']  # use C2 (green) for annotation

# Extended-c parameters
alpha    = 1.0
K        = 1/0.5
c0_ref   = -2.0
beta_b   = 0.5
beta_c   = 0.0

# Analytic E[T] under extended-c softmax
def analytic_ET(c0, beta, s, N=500):
    c_eff = c0 + K*(1 - alpha)
    t     = np.arange(1, N+1)
    g     = s * np.exp(-lam * t)
    logits= c_eff + beta * g
    pL    = 1/(1 + np.exp(logits))
    surv  = np.concatenate(([1.], np.cumprod(1 - pL)[:-1]))
    return np.dot(t, pL * surv)

plt.rcParams.update({
    'axes.facecolor':'white',
    'axes.grid':True,
    'grid.color':'lightgray',
    'grid.linestyle':'-',
    'grid.linewidth':0.5,
})

# Panel a) sweep β at c₀ = –2
betas = np.logspace(-2, 0, 200)
fig, ax = plt.subplots()
for s, col, lab in zip(patches, colors, labels):
    et = [analytic_ET(c0_ref, b, s) for b in betas]
    ax.semilogx(betas, et, color=col, lw=2, label=lab)

ax.set_xlim(1e-2,1e0)
ax.set_xticks([1e-2,1e-1,1e0])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('β (higher = exploit)')
ax.set_ylabel('Expected leaving time (s)')
ax.set_title('a) Berry: sweep over β (c₀ = –2)')
ax.legend(loc='upper left')
ax.text(0.95,0.05,f'c₀ = {c0_ref}', transform=ax.transAxes,
        color=colors[1], ha='right', va='bottom',
        fontsize=10, fontweight='bold')

# Panel b) sweep c₀ at β = 0.5
cs = np.linspace(-3, 3, 200)
fig, ax = plt.subplots()
for s, col, lab in zip(patches, colors, labels):
    et = [analytic_ET(c, beta_b, s) for c in cs]
    ax.plot(cs, et, color=col, lw=2, label=lab)

ax.set_xlim(-3,3)
ax.set_xticks(np.arange(-3,4,1))
ax.set_xlabel('c₀ (higher = exploit)')
ax.set_ylabel('Expected leaving time (s)')
ax.set_title('b) Berry: sweep over c₀ (β = 0.5)')
ax.legend(loc='upper left')
ax.text(0.95,0.05,f'β = {beta_b}', transform=ax.transAxes,
        color=colors[1], ha='right', va='bottom',
        fontsize=10, fontweight='bold')

# Panel c) sweep c₀ at β = 0
fig, ax = plt.subplots()
for s, col, lab in zip(patches, colors, labels):
    et = [analytic_ET(c, beta_c, s) for c in cs]
    ax.plot(cs, et, color=col, lw=2, label=lab)

ax.set_xlim(-3,3)
ax.set_xticks(np.arange(-3,4,1))
ax.set_xlabel('c₀ (higher = exploit)')
ax.set_ylabel('Expected leaving time (s)')
ax.set_title('c) Berry: sweep over c₀ (β = 0)')
ax.legend(loc='upper left')
ax.text(0.95,0.05,f'β = {beta_c}', transform=ax.transAxes,
        color=colors[1], ha='right', va='bottom',
        fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

