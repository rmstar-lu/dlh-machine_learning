#!/usr/bin/env python3
"""
A function that finds the best number of clusters for a GMM using the BIC
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ Find the best number of clusters for a GMM using the BIC """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(kmin) is not int or kmin < 1 or kmin >= len(X)
            or kmax is not None
            and (type(kmax) is not int or kmax <= kmin or kmax > len(X))
            or type(iterations) is not int or iterations < 1
            or type(tol) is not float or tol <= 0
            or type(verbose) is not bool):
        return 4 * (None,)

    n, d = X.shape
    if kmax is None:
        kmax = n
    best_BIC = None
    logLs = []
    BICs = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, logL = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        p = k * (d + d * (d + 1) / 2) + k - 1
        BIC = p * np.log(n) - 2 * logL
        if best_BIC is None or BIC < best_BIC:
            best_k = k
            best_result = (pi, m, S)
            best_BIC = BIC
        logLs.append(logL)
        BICs.append(BIC)
    return (best_k, best_result, np.array(logLs), np.array(BICs))
