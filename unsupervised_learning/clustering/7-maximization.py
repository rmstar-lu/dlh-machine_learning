#!/usr/bin/env python3
"""
A function that calculates the maximization step in the EM algorithm for a GMM
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

    G = g.sum(axis=1)
    pi = G / G.sum()
    m = (g @ X) / G[:, None]
    S = np.array([((g[i, :, np.newaxis] * (X - m[i])).T @ (X - m[i]))
                    for i in range(len(g))]) / G[:, None, None]
    return (pi, m, S)
