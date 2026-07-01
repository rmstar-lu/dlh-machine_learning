#!/usr/bin/env python3
""" 5-definiteness module - calculate the definiteness of a numpy matrix """

import numpy as np


def definiteness(matrix):
    """ Return a string describing the definiteness of a numpy matrix. """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    try:
        ev = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.array_equal(matrix.T, matrix):
        return None
    if all(ev_i > 0 for ev_i in ev):
        return "Positive definite"
    if all(ev_i >= 0 for ev_i in ev):
        return "Positive semi-definite"
    if all(ev_i < 0 for ev_i in ev):
        return "Negative definite"
    if all(ev_i <= 0 for ev_i in ev):
        return "Negative semi-definite"
    return "Indefinite"
