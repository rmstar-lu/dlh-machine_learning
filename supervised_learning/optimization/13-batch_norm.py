#!/usr/bin/env python3
"""
A function that normalizes an unactivated output of a neural network using
batch normalization.

Z is a numpy.ndarray of shape (m, n) that should be normalized
m is the number of data points
n is the number of features in Z
gamma is a numpy.ndarray of shape (1, n) containing the scales used for
batch normalization
beta is a numpy.ndarray of shape (1, n) containing the offsets used for
batch normalization
epsilon is a small number used to avoid division by zero

Returns: the normalized Z matrix
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalize un unactivated output using batch normalization. """
    Z_norm = (Z - Z.mean(axis=0)) / np.sqrt(Z.var(axis=0) + epsilon)
    return gamma * Z_norm + beta
