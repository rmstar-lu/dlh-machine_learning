#!/usr/bin/env python3
"""
A function that initializes variables for a Gaussian Mixture Model
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
