import numpy as np

# 1) Task-Parameter
Y = np.array([32.5, 45.0, 57.5])   # initial yields (low, mid, high)
lam = 0.075                        # decay rate

# 2) BRR-Werte (gemäss Le Heron)
BRR_rich = 21.8710
BRR_poor = 18.5684

# 3) MVT-optimale t* (wenn g(t)=BRR)
#    t* = -ln(BRR / Y) / lam
t_opt_rich = -np.log(BRR_rich / Y) / lam
t_opt_poor = -np.log(BRR_poor / Y) / lam

# 4) Ausgabe
for env, t_opt in [('Rich', t_opt_rich), ('Poor', t_opt_poor)]:
    print(f"{env}-Environment optimal t* (s):")
    for yi, t in zip(Y, t_opt):
        print(f"  Y={yi:4.1f}: t* = {t:5.2f} s")
