from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad

def weibull(s_range, lambda_, k):
    return (k/lambda_) * (s_range/lambda_)**(k-1) * np.exp(-(s_range/lambda_)**k)

def p_inf_s(s):
    pi_s = max(0, min(0.5, 0.5 - 0.08* (s)))
    return pi_s

def p_inf_b(b):
    pi_b = max(0, min(0.5, 0.5 - 0.08* (s)))
    return pi_b


def utility(bid, ask, p0, lambda_, k):
    profit = p_inf_s(p0-bid) * (p0-bid) + p_inf_b(ask-p0) * (ask-p0)
    lls, _ = quad(lambda s_range: (bid-p0) * weibull(s_range=s_range, lambda_ = lambda_, k=k), 0, bid)
    llb, _ = quad(lambda s_range: (p0- ask) * weibull(s_range=s_range, lambda_ = lambda_, k=k), ask, np.inf)

    loss = p_inf_s(bid) * lls + p_inf_b(ask) * llb
    utility = profit-loss
    return -utility

