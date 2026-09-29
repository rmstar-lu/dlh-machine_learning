#!/usr/bin/env python3
"""
A function that performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """ Perform PCA on a dataset """
    n, d = X.shape
    U, S, Vt = np.linalg.svd(X)     # full_matrices=False
    W = Vt.T
    # squared singular values are the eigenvalues
    # each eigenvalue is proportional to the variance
    eig = (S ** 2) / n - 1
    nd = np.searchsorted(np.cumsum(eig), var * eig.sum()) + 1
    return W[:, :nd + 1]            # why + 1???
