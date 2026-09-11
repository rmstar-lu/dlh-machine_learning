#!/usr/bin/env python3
"""
A function that calculates the pdf of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """ Calculate the pdf of a Gaussian distribution """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
        or not isinstance(m, np.ndarray) or len(m.shape) != 1
            or m.shape[0] != X.shape[1]
        or not isinstance(S, np.ndarray) or len(S.shape) != 2
            or S.shape[0] != X.shape[1] or S.shape[1] != X.shape[1]):
        return None

    n, d = X.shape
    v = X - m
    det = np.linalg.det(S)
    maha = ((v @ np.linalg.inv(S)) * v).sum(axis=1)
    pdf = (1. / np.exp(.5 * maha)) * (det * (2 * np.pi) ** d) ** -.5
    np.maximum(pdf, 1e-300, out=pdf)
    return pdf
