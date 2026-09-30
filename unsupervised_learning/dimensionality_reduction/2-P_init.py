#!/usr/bin/env python3
"""
A function that initializes all variables required to calculate the
P affinities in t-SNE
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initialize all variables required to calculate the P affinities in t-SNE
    """
    n, d = X.shape
    sq_norms = np.sum(X ** 2, axis=1)
    gram = X @ X.T
    D = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * gram
    D = np.maximum(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log(perplexity) / np.log(2.)
    return (D, P, betas, H)
