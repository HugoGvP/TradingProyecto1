import numpy as np
from scipy.integrate import quad
from scipy.stats import weibull_min as wbl, weibull_min
import matplotlib.pyplot as plt
from scenario1 import bid_ask
from scenario2 import revenue_scenario2 as r2

lambda_val = 50
k_shape = 10
s_range = np.linspace(0, 10, 100)

funct = bid_ask()

prices = weibull_min.rvs(k_shape, scale=lambda_val, size=100)

plt.figure(figsize=(8, 5))
plt.plot(s_range, funct, label="Liquidity Motivated Trades (Pi_I = 0)", linestyle="--")
plt.xlabel("S")
plt.ylabel("Expected Revenue")
plt.legend()
plt.title("Expected Revenue under Different Trading Scenarios")
plt.show()
