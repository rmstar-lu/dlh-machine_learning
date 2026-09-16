#!/usr/bin/env python3
"""
A function that finds the best number of clusters for a GMM using the Bayesian Information Criterion:

X is a numpy.ndarray of shape (n, d) containing the data set
kmin is a positive integer containing the minimum number of clusters to check for (inclusive)
kmax is a positive integer containing the maximum number of clusters to check for (inclusive)
    If kmax is None, kmax should be set to the maximum number of clusters possible
    iterations is a positive integer containing the maximum number of iterations for the EM algorithm
tol is a non-negative float containing the tolerance for the EM algorithm
verbose is a boolean that determines if the EM algorithm should print information to the standard output

Returns: best_k, best_result, l, b, or None, None, None, None on failure
    best_k is the best value for k based on its BIC
    best_result is tuple containing pi, m, S
        pi is a numpy.ndarray of shape (k,) containing the cluster priors for the best number of clusters
        m is a numpy.ndarray of shape (k, d) containing the centroid means for the best number of clusters
        S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for the best number of clusters
    l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood for each cluster size tested
    b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested
        Use: BIC = p * ln(n) - 2 * l
        p is the number of parameters required for the model
        n is the number of data points used to create the model
        l is the log likelihood of the model
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ Find the best number of clusters for a GMM using the Bayesian Information Criterion """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(kmin) is not int or kmin < 1
            or kmax is not None and (type(kmax) is not int or kmax <= kmin)
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
        # calculate the number of parameters p:
        # each cluster has d means + d * (d + 1) / 2 variances and covariances
        # also add the cluster weights (k - 1 because they add up to 1)
        p = k * (d + d * (d + 1) / 2) + k - 1
        # BIC penalizes larger number of parameters, but rewards higher likelihood
        # lower BIC is better
        BIC = p * np.log(n) - 2 * logL
        if best_BIC is None or BIC < best_BIC:
            best_k = k
            best_result = (pi, m, S)
            best_BIC = BIC
        logLs.append(logL)
        BICs.append(BIC)
    return (best_k, best_result, np.array(logLs), np.array(BICs))
