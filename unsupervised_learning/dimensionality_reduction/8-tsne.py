#!/usr/bin/env python3
"""
A function that performs a t-SNE transformation.
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
    for t in range(iterations):
        P = P_exaggerated if t < 100 else P_normal
        dY, Q = grads(Y, P)
        if t % 100 == 99:
            print(f"Cost at iteration {t + 1}: {cost(P, Q)}")
        alpha = .5 if t < 20 else .8
        update = alpha * update - lr * dY
        Y += update
        Y -= Y.mean(axis=0)
    return Y
