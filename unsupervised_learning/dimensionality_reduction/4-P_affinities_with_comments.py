#!/usr/bin/env python3
"""
Write a function def P_affinities(X, tol=1e-5, perplexity=30.0): that calculates the symmetric P affinities of a data set:

X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
    n is the number of data points
    d is the number of dimensions in each point
perplexity is the perplexity that all Gaussian distributions should have
tol is the maximum tolerance allowed (inclusive) for the difference in Shannon entropy from perplexity for all Gaussian distributions

Returns: P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities

Hint 1: See page 6 of t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
Hint 2: For this task, you will need to perform a binary search on each point's distribution to find the correct value of beta that will give a Shannon Entropy H within the tolerance (Think about why we analyze the Shannon entropy instead of perplexity). Since beta can be in the range (0, inf), you will have to do a binary search with the high and low initially set to None. If in your search, you are supposed to increase/decrease beta to high/low but they are still set to None, you should double/half the value of beta instead.
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
