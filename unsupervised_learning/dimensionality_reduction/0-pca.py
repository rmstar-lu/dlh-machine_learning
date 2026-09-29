#!/usr/bin/env python3
"""
A function that performs PCA on a dataset:

X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
var is the fraction of the variance that the PCA transformation should maintain

Returns: the weights matrix, W, that maintains var fraction of X's original variance
W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of the transformed X
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
