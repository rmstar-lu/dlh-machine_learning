#!/usr/bin/env python3
"""
Write a function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation:

X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
ndims is the new dimensional representation of X
idims is the intermediate dimensional representation of X after PCA
perplexity is the perplexity
iterations is the number of iterations
lr is the learning rate

Every 100 iterations, not including 0, print Cost at iteration {iteration}: {cost}
{iteration} is the number of times Y has been updated and {cost} is the corresponding cost
After every iteration, Y should be re-centered by subtracting its mean

Returns: Y, a numpy.ndarray of shape (n, ndim) containing the optimized low dimensional transformation of X

For the first 100 iterations, perform early exaggeration with an exaggeration of 4
a(t) = 0.5 for the first 20 iterations and 0.8 thereafter

Hint 1: See Algorithm 1 on page 9 of t-SNE. But WATCH OUT! There is a mistake in the gradient descent step
Hint 2: See Section 3.4 starting on page 9 of t-SNE for early exaggeration
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """ Perform a  t-SNE transformation """
    X = pca(X, idims)
    P_normal = P_affinities(X, perplexity=perplexity)
    P_exaggerated = P_normal * 4
    Y = np.random.randn(X.shape[0], ndims)
    update = np.zeros(Y.shape)
    """
    gains = np.ones(Y.shape)
    """
    for t in range(iterations):
        P = P_exaggerated if t < 100 else P_normal
        dY, Q = grads(Y, P)
        if t % 100 == 99:
            print(f"Cost at iteration {t + 1}: {cost(P, Q)}")
        alpha = .5 if t < 20 else .8
        """
        inc = (update * dY) < 0.0   # different sign
        dec = np.invert(inc)        # equal sign
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, .01, np.inf, out=gains)
        dY *= gains
        """
        update = alpha * update - lr * dY
        Y += update
        Y -= Y.mean(axis=0)
    return Y
