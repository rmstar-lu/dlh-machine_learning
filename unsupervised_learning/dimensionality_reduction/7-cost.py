#!/usr/bin/env python3
"""
A function that calculates the cost of the t-SNE transformation.
"""
import numpy as np


def cost(P, Q):
    """ Calculate the cost of the t-SNE transformation """
    return (P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12))).sum()
