#!/usr/bin/env python3
"""
A function that performs the expectation maximization for a GMM:
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ Perform the expectation maximization for a GMM """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(k) is not int or k < 1
            or type(iterations) is not int or iterations < 1
            or type(tol) is not float or tol <= 0
            or type(verbose) is not bool):
        return 5 * (None,)

    pi, m, S = initialize(X, k)
    logL_prev = np.nan
    for i in range(iterations):
        g, logL = expectation(X, pi, m, S)
        if abs(logL - logL_prev) <= tol:
            iterations = i
            break
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {logL:.5f}")
        pi, m, S = maximization(X, g)
        logL_prev = logL
    if verbose:
        print(f"Log Likelihood after {iterations} iterations: {logL:.5f}")
    return (pi, m, S, g, logL)
