#!/usr/bin/env python3
"""
Module 1-intersection, calculate the intersection of obtaining given data
with the hypothetical probabilities, i.e.
P(B|A)*P(A) in Bayes' rule P(A|B) = P(B|A)*P(A)/P(B)
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    x is the number of patients that develop severe side effects
    (binomial distribution)
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the hypothetical probabilities
    of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than "
                         "or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1.):
        raise ValueError("Pr must sum to 1")

    n_choose_x = 1
    for i in range(x + 1, n + 1):
        n_choose_x *= i
    for i in range(2, n - x + 1):
        n_choose_x //= i
    return float(n_choose_x) * P ** x * (1 - P) ** (n - x) * Pr
