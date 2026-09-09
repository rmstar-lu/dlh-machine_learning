#!/usr/bin/env python3
"""
A function that tests for the optimum number of clusters by variance
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ Test for the optimum number of clusters by variance """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(kmin) is not int or kmin <= 0
            or type(kmax) is not int or kmax <= kmin
            or type(iterations) is not int or iterations <= 0):
        return (None, None)

    results, variances = [], []
    for k in range(kmin, kmax + 1):
        C, labels = kmeans(X, k, iterations)
        results.append((C, labels))
        var = variance(X, C)
        print(f"k = {k}, var = {var}")
        variances.append(var)
    return (results, list(variances[0] - np.array(variances)))
