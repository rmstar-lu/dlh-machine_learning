#!/usr/bin/env python3
"""
A function that calculates the expectation step in the EM algorithm for a GMM:

X is a numpy.ndarray of shape (n, d) containing the data set
pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
You may use at most 1 loop
Returns: g, l, or None, None on failure
g is a numpy.ndarray of shape (k, n) containing the posterior probabilities for each data point in each cluster
l is the total log likelihood
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
            or S.shape[0] != len(pi) or S.shape[1] != X.shape[1] or S.shape[2] != X.shape[1]):
        return (None, None)

    N, d = X.shape
    k = len(pi)
    # for all i from 1 to k:
    #    for all n from 1 to N:
    #       calulate P(z_n = i|x_n) = pi_i*Norm(x_n|m_i, S_i)/sum_j=1^k pi_j*Norm(x_n|m_j, S_j)
    #       this is the posterior probability for x_n belonging to cluster i
    L = np.array([pdf(X, m[i], S[i]) for i in range(k)])
    M = pi @ L
    P = (pi[:, np.newaxis] * L) / M
    return (P, np.log(M).sum())
