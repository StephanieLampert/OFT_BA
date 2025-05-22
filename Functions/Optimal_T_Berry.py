import numpy as np

# 1) Berry‐Task Parameter
Y = np.array([34.5, 57.5])    # low, high yields
lam = 0.11                    # decay rate

# 2) BRR-Werte für Berry Task
BRR_rich = 23.7513
BRR_poor = 19.2604

# 3) MVT‐optimale t* (g(t) = BRR)
#    t* = -ln(BRR / Y) / lam
t_opt_rich = -np.log(BRR_rich / Y) / lam
t_opt_poor = -np.log(BRR_poor / Y) / lam

# 4) Ausgabe
for env, t_opt in [('Rich', t_opt_rich), ('Poor', t_opt_poor)]:
    print(f"{env}-Environment optimal t* (s):")
    for yi, t in zip(Y, t_opt):
        print(f"  Y={yi:4.1f}: t* = {t:5.2f} s")
