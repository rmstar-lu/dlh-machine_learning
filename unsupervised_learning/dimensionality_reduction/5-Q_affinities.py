#!/usr/bin/env python3
"""
A function that calculates the Q affinities.
"""
import numpy as np


def Q_affinities(Y):
    """ Calculate the Q affinities """
    n, ndim = Y.shape
    sq_norms = np.sum(Y ** 2, axis=1)
    gram = Y @ Y.T
    D = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * gram
    D = np.maximum(D, 0)
    num = 1. / (1. + D)     # Student t-distribution with 1 df
    np.fill_diagonal(num, 0)
    return (num / np.sum(num), num)
