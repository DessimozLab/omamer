"""
Small statistical helpers used by OMAmer search.
"""
import math

import numba
import numpy as np


@numba.njit(nogil=True)
def sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@numba.njit(nogil=True)
def beta_binomial_params_for_n(n, q_coef, kappa_coef, log_n_center, log_n_scale):
    z = (math.log(float(n)) - log_n_center) / log_n_scale

    q_linear = 0.0
    z_power = 1.0
    for i in range(q_coef.size):
        q_linear += q_coef[i] * z_power
        z_power *= z
    q = min(max(sigmoid(q_linear), 1e-8), 1.0 - 1e-8)

    log_kappa = 0.0
    z_power = 1.0
    for i in range(kappa_coef.size):
        log_kappa += kappa_coef[i] * z_power
        z_power *= z
    log_kappa = min(max(log_kappa, math.log(1e-4)), math.log(1e8))
    kappa = math.exp(log_kappa)

    alpha = q * kappa
    beta = (1.0 - q) * kappa
    return alpha, beta, q


@numba.njit(nogil=True)
def beta_binomial_logpmf(x, n, alpha, beta):
    return (
        math.lgamma(n + 1.0)
        - math.lgamma(x + 1.0)
        - math.lgamma(n - x + 1.0)
        + math.lgamma(x + alpha)
        + math.lgamma(n - x + beta)
        - math.lgamma(n + alpha + beta)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
    )


@numba.njit(nogil=True)
def beta_binomial_neglogccdf(x, n, alpha, beta):
    if x <= 0:
        return 0.0
    if x > n:
        return np.inf

    max_logp = -np.inf
    for k in range(x, n + 1):
        logp = beta_binomial_logpmf(float(k), float(n), alpha, beta)
        if logp > max_logp:
            max_logp = logp

    acc = 0.0
    for k in range(x, n + 1):
        acc += math.exp(beta_binomial_logpmf(float(k), float(n), alpha, beta) - max_logp)

    return -(max_logp + math.log(acc))
