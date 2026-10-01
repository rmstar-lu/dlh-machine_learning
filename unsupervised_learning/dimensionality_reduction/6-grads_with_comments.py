#!/usr/bin/env python3
"""
Write a function def grads(Y, P): that calculates the gradients of Y:

Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
P is a numpy.ndarray of shape (n, n) containing the P affinities of X

Do not multiply the gradients by the scalar 4 as described in the paper's equation

Returns: (dY, Q)
    dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
    Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y

Hint: See page 8 of t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Calculate the gradients of Y """
    Q, num = Q_affinities(Y)
    # scaling has shape (n, n)
    scaling = (P - Q) * num
    # pairwise differences of vectors in Y, shape (n, n, ndim) 
    diff = Y[:, None] - Y[None, :]
    # scaling[:, :, None] has shape (n, n, 1)
    # sum over j to get shape (n, ndim)
    dY = np.sum(scaling[:, :, None] * diff, axis=1)
    return (dY, Q)
