from stat import UF_IMMUTABLE

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import bid_ask as ba
from functools import partial


lambda_ = 50
k = 10
s_range = np.linspace(0, 100, 1000)
s = np.linspace(0,10,100)
p0 = 51
pi_inf = 0.4
pi_lb = 0.5
pi_ls = 0.5


bid = p0 - 5
ask = p0 + 5


funct = ba.weibull(s_range, lambda_, k)

plt.figure(figsize=(8, 5))
plt.plot(s_range, funct, label="Liquidity Motivated Trades (Pi_I = 0)", color="blue")
plt.xlabel("S")
plt.ylabel("Expected Revenue")
plt.legend()
plt.title("Expected Revenue under Different Trading Scenarios")
plt.show()


initial_params = [p0 - p0*0.15, p0 + p0*0.15]

# Define the objective function with fixed parameters
utility_func = partial(ba.utility, p0=p0, lambda_=lambda_, k=k)

# Optimize the bid-ask spread
result = minimize(lambda x: utility_func(x[0], x[1]), initial_params, bounds=[(0, p0), (p0, 5 * p0)])
bid_opt, ask_opt = result.x

print(f"Optimal Bid: {bid_opt:.4f}")
print(f"Optimal Ask: {ask_opt:.4f}")
print(f"Optimal Spread: {ask_opt - bid_opt:.4f}")











