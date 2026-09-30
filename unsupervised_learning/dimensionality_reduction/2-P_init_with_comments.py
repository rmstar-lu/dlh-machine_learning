#!/usr/bin/env python3
"""
Write a function def P_init(X, perplexity): that initializes all variables required to calculate the P affinities in t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
perplexity is the perplexity that all Gaussian distributions should have

Returns: (D, P, betas, H)
    D: a numpy.ndarray of shape (n, n) that calculates the squared pairwise distance between two data points
    The diagonal of D should be 0s
    P: a numpy.ndarray of shape (n, n) initialized to all 0's that will contain the P affinities
    betas: a numpy.ndarray of shape (n, 1) initialized to all 1's that will contain all of the beta values
    \beta_{i} = \frac{1}{2{\sigma_{i}}^{2} }
    H is the Shannon entropy for perplexity perplexity with a base of 2
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initialize all variables required to calculate the P affinities in t-SNE
    """
    n, d = X.shape
    # Calculate squared distances:
    sq_norms = np.sum(X ** 2, axis=1)   # row-wise squared norm (n,)
    gram = X @ X.T                      # Gram matrix G_ij == x_i @ x_j
    # D = norm(x - y)^2 == norm(x)^2 + norm(y)^2 - 2xy
    D = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * gram
    D = np.maximum(D, 0)                # Clip to 0 in case of rounding errors
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log(perplexity) / np.log(2.) # Shannon entropy using base 2
    return (D, P, betas, H)
