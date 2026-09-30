#!/usr/bin/env python3
"""
Write a function def HP(Di, beta): that calculates the Shannon entropy and P affinities relative to a data point:

Di is a numpy.ndarray of shape (n - 1,) containing the pariwise distances between a data point and all other points except itself
    n is the number of data points
beta is a numpy.ndarray of shape (1,) containing the beta value for the Gaussian distribution

Returns: (Hi, Pi)
    Hi: the Shannon entropy of the points
    Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities of the points

Hint: see page 4 of t-SNE (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
"""
import numpy as np


def HP(Di, beta):
    """ Calculate the Shannon entropy and P affinities relative to a data point """
    Pi = np.exp(-Di * beta)
    Pi = Pi / Pi.sum()
    Hi = -np.sum(np.where(Pi > 0, Pi * np.log2(Pi), 0))
    return (Hi, Pi)
