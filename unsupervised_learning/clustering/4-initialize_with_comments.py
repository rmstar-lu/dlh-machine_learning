#!/usr/bin/env python3
"""
A function that initializes variables for a Gaussian Mixture Model:

X is a numpy.ndarray of shape (n, d) containing the data set
k is a positive integer containing the number of clusters
You are not allowed to use any loops
Returns: pi, m, S, or None, None, None on failure
pi is a numpy.ndarray of shape (k,) containing the priors for each cluster,
initialized evenly
m is a numpy.ndarray of shape (k, d) containing the centroid means for each
cluster, initialized with K-means
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
for each cluster, initialized as identity matrices
"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ Initialize variables for a Gaussian Mixture Model """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(k) is not int or k <= 0):
        return (None, None, None)

    n, d = X.shape
    C, labels = kmeans(X, k)
    return (np.full(k, 1/k), C, np.array(k * [np.eye(d)]))
