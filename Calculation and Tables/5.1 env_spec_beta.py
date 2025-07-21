# Die hier verwendeten Werte stammen direkt aus 1.1 BRR_Calculation.py

# 1) BRR‐Werte manuell festlegen:
milk_rich_brr  = 21.8710
milk_poor_brr  = 18.5684
berry_rich_brr = 23.7513
berry_poor_brr = 19.2604

# 2) Meta‐Parameter setzen:
gamma_beta  = -0.5
lambda_beta =  2.3

# 3) Umgebungsabhängige β‐Werte berechnen:
beta_milk_rich  = lambda_beta * (milk_rich_brr  ** gamma_beta)
beta_milk_poor  = lambda_beta * (milk_poor_brr  ** gamma_beta)
beta_berry_rich = lambda_beta * (berry_rich_brr ** gamma_beta)
beta_berry_poor = lambda_beta * (berry_poor_brr ** gamma_beta)

# 4) Ergebnisse ausgeben:
print(f"Milk Task – Rich Environment:  β ≈ {beta_milk_rich:.4f}")
print(f"Milk Task – Poor Environment:  β ≈ {beta_milk_poor:.4f}")
print(f"Berry Task – Rich Environment: β ≈ {beta_berry_rich:.4f}")
print(f"Berry Task – Poor Environment: β ≈ {beta_berry_poor:.4f}")

