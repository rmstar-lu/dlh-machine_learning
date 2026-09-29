#!/usr/bin/env python3
"""
A function that performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """ Perform PCA on a dataset """
    U, S, _ = np.linalg.svd(X)     # full_matrices=False
    return U[:, :ndim] @ np.diag(S[:ndim])
