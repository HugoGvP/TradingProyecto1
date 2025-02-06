from scipy.integrate import quad

def revenue_scenario2(a, b, p0, lb, ls, i ,pdf_values):
    cost_ask, _ = quad(lambda p0: (p0 - a) * pdf_values, a, np.inf)
    cost_bid, _ = quad(lambda p0: (b - p0) * pdf_values, 0, b)
    r2 = (a - p0)*lb + (p0 - b)*ls - (cost_ask+cost_bid)*i
    return r2