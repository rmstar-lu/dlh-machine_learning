#!/usr/bin/env python3
"""
A function that calculates the Q affinities.
"""
import numpy as np


def Q_affinities(Y):
    """ Calculate the Q affinities """
    D = ((Y[:, None] - Y[None, :]) ** 2).sum(axis=2)
    num = 1. / (1. + D)     # Student t-distribution with 1 df
    np.fill_diagonal(num, 0)
    return (num / np.sum(num), num)
