#!/usr/bin/env python3
"""
A function that calculates the Shannon entropy and P affinities relative
to a data point
"""
import numpy as np


def HP(Di, beta):
    """
    Calculate the Shannon entropy and P affinities relative to a data point
    """
    if len(Di) == 0:
        return (0., Di.copy())
    Pi = np.exp(-Di / (beta[0] if isinstance(beta, np.ndarray) else beta))
    Pi = Pi / Pi.sum()
    Hi = -np.sum(Pi * np.log(Pi)) / np.log(2.)
    return (Hi, Pi)
