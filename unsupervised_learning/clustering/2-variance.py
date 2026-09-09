#!/usr/bin/env python3
"""
A function that calculates the total intra-cluster variance for a data set
"""
import numpy as np


def variance(X, C):
    """ Calculate the total intra-cluster variance for a data set """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or not isinstance(C, np.ndarray) or len(C.shape) != 2
            or C.shape[0] < 1 or C.shape[1] != X.shape[1]):
        return None

    n, d = X.shape
    k = len(C)
    dist_sq = ((X[:, np.newaxis, :] - C) ** 2).sum(axis=2)
    labels = np.argmin(dist_sq, axis=1)
    sum_dist_sq = np.zeros(k)
    np.add.at(sum_dist_sq, labels, dist_sq[np.arange(n), labels])
    return sum_dist_sq.sum()
