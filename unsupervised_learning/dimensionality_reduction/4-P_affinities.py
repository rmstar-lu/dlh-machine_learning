#!/usr/bin/env python3
"""
A function that calculates the symmetric P affinities of a data set.
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """ Calculate the symmetric P affinities of a data set """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(len(X)):
        high, low = None, None
        while True:
            Hi, Pi = HP(np.concatenate((D[i, :i], D[i, i + 1:])), betas[i])
            if abs(Hi - H) <= tol:
                break
            if Hi < H:
                # sigma is too low, beta is too high
                high = betas[i, 0]
            else:
                # sigma is too high, beta is too low
                low = betas[i, 0]
            if low is None:
                betas[i] = high / 2
            elif high is None:
                betas[i] = low * 2
            else:
                betas[i] = low + (high - low) / 2
        P[i, :i], P[i, i], P[i, i + 1:] = Pi[:i], 0., Pi[i:]
    return (P + P.T) / (2 * n)
