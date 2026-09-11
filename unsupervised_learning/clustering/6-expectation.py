#!/usr/bin/env python3
"""
A function that calculates the expectation step in the EM algorithm for a GMM
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ Calculates the expectation step in the EM algorithm for a GMM """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
        or not isinstance(pi, np.ndarray) or len(pi.shape) != 1
            or len(pi) < 1 or not np.allclose(pi.sum(), 1.)
        or not isinstance(m, np.ndarray) or len(m.shape) != 2
            or m.shape[0] != len(pi) or m.shape[1] != X.shape[1]
        or not isinstance(S, np.ndarray) or len(S.shape) != 3
            or S.shape[0] != len(pi)
            or S.shape[1] != X.shape[1] or S.shape[2] != X.shape[1]):
        return (None, None)

    N, d = X.shape
    k = len(pi)
    L = np.array([pdf(X, m[i], S[i]) for i in range(k)])
    M = pi @ L
    P = (pi[:, np.newaxis] * L) / M
    return (P, np.log(M).sum())
