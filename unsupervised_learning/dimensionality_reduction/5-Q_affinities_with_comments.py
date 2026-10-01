#!/usr/bin/env python3
"""
Write a function def Q_affinities(Y): that calculates the Q affinities:

Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
    n is the number of points
    ndim is the new dimensional representation of X

Returns: Q, num
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    num is a numpy.ndarray of shape (n, n) containing the numerator of the Q affinities

Hint: See page 7 of t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
"""
import numpy as np


def Q_affinities(Y):
    """ Calculate the Q affinities """
    D = ((Y[:, None] - Y[None, :]) ** 2).sum(axis=2)
    num = 1. / (1. + D)     # Student t-distribution with 1 df
    np.fill_diagonal(num, 0)
    return (num / np.sum(num), num)
