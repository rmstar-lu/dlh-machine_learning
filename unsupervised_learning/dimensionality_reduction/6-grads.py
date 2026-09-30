#!/usr/bin/env python3
"""
A function that calculates the gradients of Y
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ Calculate the gradients of Y """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    scaling = (P - Q) * num
    diff = Y[:, None] - Y[None, :]
    dY = np.sum(scaling[:, :, None] * diff, axis=1)
    return (dY, Q)
