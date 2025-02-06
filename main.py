import numpy as np
from scipy.integrate import quad
from scipy.stats import weibull_min as wbl, weibull_min
import matplotlib.pyplot as plt
from scenario1 import revenue_scenario1 as r1
from scenario2 import revenue_scenario2 as r2

lambda_val = 50
k_shape = 10
lb = 0.5
ls = 0.5
a = 53
b = 48
p0 = 51
P0 = 51
pi_inf = 0.4



weibull_dist = wbl(c=k_shape, scale=lambda_val)

x = np.linspace(0.001, 0.5, 100)

pdf_values = weibull_dist.pdf(x)

revenue = r1(a, b, P0, lb, ls)

print("Ganancia del market maker: " , revenue)

weibull_dist = weibull_min(c=k_shape, scale=lambda_val)

def E_above(threshold):
    num, _ = quad(lambda v: v * weibull_dist.pdf(v), threshold, np.inf)
    denom = 1 - weibull_dist.cdf(threshold)
    if denom > 0:
        return num / denom
    else:
        return threshold

def E_below(threshold):
    num, _ = quad(lambda v: v * weibull_dist.pdf(v), 0, threshold)
    denom = weibull_dist.cdf(threshold)
    if denom > 0:
        return num / denom
    else:
        return threshold

def adverse_selection_cost(S, P0, pi_inf):
    threshold_ask = P0 + S
    prob_ask = 1 - weibull_dist.cdf(threshold_ask)
    exp_above = E_above(threshold_ask)
    loss_ask = exp_above - threshold_ask

    threshold_bid = P0 - S
    prob_bid = weibull_dist.cdf(threshold_bid)
    exp_below = E_below(threshold_bid)
    loss_bid = threshold_bid - exp_below

    cost = (pi_inf / 2) * (prob_ask * loss_ask + prob_bid * loss_bid)
    return cost

def revenue_scenario2(S, P0=P0, pi_inf=pi_inf):
    liquidity_revenue = S  # (0.5 + 0.5) * S = S
    cost = adverse_selection_cost(S, P0, pi_inf)
    return liquidity_revenue - cost


S_vals = np.linspace(0.001, 0.5, 100)
revenue2_vals = np.array([revenue_scenario2(s) for s in S_vals])


plt.figure(figsize=(8, 5))
plt.plot(x, pdf_values, lw=2, color='blue')
plt.plot(S_vals, revenue2_vals, lw=2, color='blue', label="Revenue escenario 2")
plt.xlabel("Spread S")
plt.ylabel("Expected Revenue")
plt.title("Escenario 2: Revenue con π_I = 0.4 y liquidez fija (0.5 por lado)")
plt.grid(True)
plt.legend()
plt.show()