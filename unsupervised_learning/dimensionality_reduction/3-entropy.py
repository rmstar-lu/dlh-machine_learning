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
    Pi = np.exp(-Di * beta)
    Pi = Pi / Pi.sum()
    Hi = -np.sum(np.where(Pi > 0, Pi * np.log2(Pi), 0))
    return (Hi, Pi)
