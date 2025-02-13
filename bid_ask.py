from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
from functools import partial

def weibull(s_range, lambda_, k):
    """ Calcula la distribución Weibull """
    return (k / lambda_) * (s_range / lambda_)**(k - 1) * np.exp(-(s_range / lambda_)**k)

def p_inf_s(s):
    """ Calcula la probabilidad informada p_inf_s con soporte de arrays NumPy """
    return np.maximum(0, np.minimum(0.5, 0.5 - 0.08 * s))

def p_inf_b(b):
    """ Calcula la probabilidad informada p_inf_b con soporte de arrays NumPy """
    return np.maximum(0, np.minimum(0.5, 0.5 - 0.08 * b))

def utility(bid, ask, p0, lambda_, k):
    """ Calcula la utilidad esperada para optimización del bid-ask spread """
    profit = p_inf_s(p0 - bid) * (p0 - bid) + p_inf_b(ask - p0) * (ask - p0)

    lls, _ = quad(lambda s_range: (bid - p0) * weibull(s_range, lambda_, k), 0, bid)
    llb, _ = quad(lambda s_range: (p0 - ask) * weibull(s_range, lambda_, k), ask, np.inf)

    loss = p_inf_s(bid) * lls + p_inf_b(ask) * llb
    return -(profit - loss)  # Se minimiza la utilidad negativa

# ---- Optimización del bid-ask spread ----
def optimize_bid_ask(p0, lambda_, k):
    """Optimiza el bid y ask spread para maximizar la utilidad."""
    initial_params = [p0 * 0.85, p0 * 1.15]  # 15% abajo y arriba de p0
    utility_func = partial(utility, p0=p0, lambda_=lambda_, k=k)

    result = minimize(lambda x: utility_func(x[0], x[1]),
                      initial_params,
                      bounds=[(0, p0), (p0, 5 * p0)])

    return result.x



