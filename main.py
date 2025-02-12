import numpy as np
from scipy.integrate import quad
from scipy.stats import weibull_min as wbl, weibull_min
import matplotlib.pyplot as plt
import bid_ask

lambda_ = 50
k = 10
s_range = np.linspace(0, 100, 1000)
p0 = 51

bid = p0 - 5
ask = p0 + 5


funct = bid_ask.weibull(s_range, lambda_, k)

plt.figure(figsize=(8, 5))
plt.plot(s_range, funct, label="Liquidity Motivated Trades (Pi_I = 0)", linestyle="--")
plt.xlabel("S")
plt.ylabel("Expected Revenue")
plt.legend()
plt.title("Expected Revenue under Different Trading Scenarios")
plt.show()

pi_inf = 0.4

