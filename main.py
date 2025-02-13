import numpy as np
import matplotlib.pyplot as plt
import bid_ask as ba

# ---- Parámetros ----
lambda_ = 50
k = 10
p0 = 51
pi_inf = 0.4

s_range = np.linspace(0, 100, 1000)
s = np.linspace(0, 10, 100)

# ---- Cálculo de la distribución Weibull ----
weibull_values = ba.weibull(s_range, lambda_, k)

# ---- Gráfica de la distribución Weibull ----
plt.figure(figsize=(8, 5))
plt.plot(s_range, weibull_values, label="Liquidity Motivated Trades (Pi_I = 0)", color="blue")
plt.xlabel("S")
plt.ylabel("Expected Revenue")
plt.legend()
plt.title("Expected Revenue under Different Trading Scenarios")
plt.show()



bid_opt, ask_opt = ba.optimize_bid_ask(p0, lambda_, k)

print(f"Optimal Bid: {bid_opt:.4f}")
print(f"Optimal Ask: {ask_opt:.4f}")
print(f"Optimal Spread: {ask_opt - bid_opt:.4f}")

# ---- Visualización de la optimización ----
plt.figure(figsize=(10, 6))
plt.plot(s_range, weibull_values, label="Weibull", color="black")
plt.axvline(p0, color='orange', linestyle='-', label=f'Spot Price: {p0}')
plt.axvline(bid_opt, color='green', linestyle='-', label=f'Optimal Bid: {bid_opt:.2f}')
plt.axvline(ask_opt, color='red', linestyle='-', label=f'Optimal Ask: {ask_opt:.2f}')
plt.xlabel("Price")
plt.ylabel("Density")
plt.title("Optimal Bid-Ask Spread")
plt.legend()
plt.show()

# ---- Cálculo de utilidad esperada ----
# Utilidad cuando es liquidez
u_l = s
# Pérdida cuando es informado
u_i = -s

ex_value_liquid = u_l * (1 - pi_inf)
ex_value_informed = s * (1 - pi_inf) * ba.p_inf_s(s)  # Operación vectorizada con NumPy

# ---- Gráfica de utilidad esperada ----
plt.figure(figsize=(10, 5))
plt.plot(s, u_l, label="Utility when liquid", color="blue")
plt.plot(s, ex_value_liquid, label="Expected Utility (Liquid)", linestyle="dashed", color="green")
plt.plot(s, ex_value_informed, label="Expected Utility (Informed)", linestyle="dotted", color="red")
plt.xlabel('Price')
plt.ylabel('Expected Value')
plt.grid(True)
plt.title('Expected Value Analysis')
plt.legend()
plt.show()


