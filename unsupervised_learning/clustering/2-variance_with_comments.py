#!/usr/bin/env python3
"""
A function that calculates the total intra-cluster variance for a data set

X is a numpy.ndarray of shape (n, d) containing the data set
C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
You are not allowed to use any loops

Returns: var, or None on failure
var is the total variance

Note that this "variance" is just the sum of the squared distances to the cluster center!
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
    # dist.sq.shape == (n, k)
    dist_sq = ((X[:, np.newaxis, :] - C) ** 2).sum(axis=2)
    labels = np.argmin(dist_sq, axis=1)
    # equivalent of + over + over each from from each :dist_sq :labels group :labels
    sum_dist_sq = np.zeros(k)
    np.add.at(sum_dist_sq, labels, dist_sq[np.arange(n), labels])
    return sum_dist_sq.sum()
