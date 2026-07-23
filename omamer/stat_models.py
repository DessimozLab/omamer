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
    """
    Neg-log of the upper tail P(X >= x) of the Beta-Binomial(n, alpha, beta).

    We anchor on the single term P(x) -- one ``lgamma``-heavy call --
    and walk the rest of the tail with the closed-form PMF ratio

        P(k+1)/P(k) = (n - k)/(k + 1) * (k + alpha)/(n - k - 1 + beta)

    which costs a handful of multiplications per term. OMAmer only evaluates
    this in the upper tail (the count filter guarantees x >= expected mean,
    i.e. x is at or above the mode), so P(x) is the largest term and factoring
    it out keeps the running sum in [1, ~few) with no overflow. Once the tail
    is decaying and the next contribution is negligible we stop early.
    """
    if x <= 0:
        return 0.0
    if x > n:
        return np.inf

    log_px = beta_binomial_logpmf(float(x), float(n), alpha, beta)

    acc = 1.0
    term = 1.0
    for k in range(x, n):
        ratio = ((n - k) / (k + 1.0)) * ((k + alpha) / (n - k - 1.0 + beta))
        term *= ratio
        acc += term
        # tail is decaying and this term no longer moves the sum -> stop
        if ratio < 1.0 and term < acc * 1e-16:
            break

    return -(log_px + math.log(acc))
