#!/usr/bin/env python3
"""
A function that performs the expectation maximization for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set

k is a positive integer containing the number of clusters

iterations is a positive integer containing the maximum number of iterations for the algorithm

tol is a non-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal to tol you should stop the algorithm

verbose is a boolean that determines if you should print information about the algorithm
    If True, print Log Likelihood after {i} iterations: {l} every 10 iterations and after the last iteration
    {i} is the number of iterations of the EM algorithm
    {l} is the log likelihood, rounded to 5 decimal places

You may use at most 1 loop

Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
    g is a numpy.ndarray of shape (k, n) containing the probabilities for each data point in each cluster
    l is the log likelihood of the model
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
