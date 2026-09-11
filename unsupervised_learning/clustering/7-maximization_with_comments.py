#!/usr/bin/env python3
"""
A function that calculates the maximization step in the EM algorithm for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities for each data point in each cluster

You may use at most 1 loop

Returns: pi, m, S, or None, None, None on failure

    pi is a numpy.ndarray of shape (k,) containing the updated priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the updated centroid means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the updated covariance matrices for each cluster
"""
import numpy as np


def maximization(X, g):
    """ Calculates the expectation step in the EM algorithm for a GMM """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
        or not isinstance(g, np.ndarray) or len(g.shape) != 2
            or g.shape[0] < 1 or g.shape[1] != X.shape[0]
            or not np.allclose(np.sum(g, axis=0), 1.0)):
        return (None, None, None)

    G = g.sum(axis=1)           # sum of posterior probabilities by cluster
    pi = G / G.sum()
    m = (g @ X) / G[:, None]    # weighted avg of X using g as weights
    # Covariance matrices are scaled by g / G
    # \Sigma_i = \frac{1}{G_i} \sum_{j=1}^n g_{ij}(x_j - m_i)^T(x_j - m_i)
    S = np.array([((g[i, :, np.newaxis] * (X - m[i])).T @ (X - m[i]))
                    for i in range(len(g))]) / G[:, None, None]
    return (pi, m, S)
