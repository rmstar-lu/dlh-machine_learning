#!/usr/bin/env python3
"""
A function that performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """ Perform PCA on a dataset """
    X = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt.T
    return X @ W[:, :ndim]
