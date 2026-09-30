#!/usr/bin/env python3
"""
Write a function def cost(P, Q): that calculates the cost of the t-SNE transformation:

P is a numpy.ndarray of shape (n, n) containing the P affinities
Q is a numpy.ndarray of shape (n, n) containing the Q affinities

Returns: C, the cost of the transformation

Hint 1: See page 5 of t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
Hint 2: Watch out for division by 0 errors! Take the minimum of all values in p and q with almost 0 (ex. 1e-12)
"""
import numpy as np


def cost(P, Q):
    """ Calculate the cost of the t-SNE transformation """
    return (P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12))).sum()
