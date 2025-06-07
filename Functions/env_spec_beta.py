# Angenommen, BRR_Calculation.py wurde bereits ausgeführt und die folgenden Werte stehen fest:
#   Milk‐Task: rich_brr ≈ 21.8710, poor_brr ≈ 18.5684
#   Berry‐Task: rich_brr ≈ 23.7513, poor_brr ≈ 19.2604

# Die hier verwendeten Werte stammen direkt aus BRR_Calculation.py :contentReference[oaicite:0]{index=0}.

# 1) BRR‐Werte manuell festlegen:
milk_rich_brr  = 21.8710
milk_poor_brr  = 18.5684
berry_rich_brr = 23.7513
berry_poor_brr = 19.2604

# 2) Meta‐Parameter setzen (wie in der Thesis):
gamma_beta  = -0.5
lambda_beta =  2.3

# 3) Umgebungs­abhängige β‐Werte berechnen:
beta_milk_rich  = lambda_beta * (milk_rich_brr  ** gamma_beta)
beta_milk_poor  = lambda_beta * (milk_poor_brr  ** gamma_beta)
beta_berry_rich = lambda_beta * (berry_rich_brr ** gamma_beta)
beta_berry_poor = lambda_beta * (berry_poor_brr ** gamma_beta)

# 4) Ergebnisse ausgeben:
print(f"Milk Task – Rich Environment:  β ≈ {beta_milk_rich:.4f}")
print(f"Milk Task – Poor Environment:  β ≈ {beta_milk_poor:.4f}")
print(f"Berry Task – Rich Environment: β ≈ {beta_berry_rich:.4f}")
print(f"Berry Task – Poor Environment: β ≈ {beta_berry_poor:.4f}")
