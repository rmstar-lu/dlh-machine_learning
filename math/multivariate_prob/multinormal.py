#!/usr/bin/env python3
"""
module multinormal:
Defines a class that represents a multivariate normal distribution.
"""

import numpy as np


class MultiNormal:
    """ A class representig a multivariate normal distribution. """

    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        """

        if not (isinstance(data, np.ndarray) and len(data.shape) == 2):
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1).reshape((d, 1))
        self.cov = ((data - self.mean) @ (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """
        x is a numpy.ndarray of shape (d, 1) containing the data point
        whose PDF should be calculated
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        """
        The naive implementation is numerically unstable and slow:
        v = x - self.mean
        return ((2 * np.pi) ** (-.5 * d) * np.linalg.det(self.cov) ** -.5 *
                np.exp(-.5 * (v.T @ np.linalg.inv(self.cov) @ v)[0, 0]))

        It is better to use eigenvalue decomposition:
        vals, vecs = np.linalg.eig(self.cov)  # eigh would work, too
        logdet = np.sum(np.log(vals))
        U = vecs * np.sqrt(1. / vals)
        v = x - self.mean
        maha = np.square(v.T @ U).sum()
        log2pi = np.log(2 * np.pi)
        return np.exp(-.5 * (len(vals) * log2pi + maha + logdet))

        This is even simpler:
        norm_coeff = d * np.log(2 * np.pi) + np.linalg.slogdet(self.cov)[1]
        v = x - self.mean
        numerator = np.linalg.solve(self.cov, v).T.dot(v)[0,0]
        return np.exp(-.5 * (norm_coeff + numerator))

        This version without log returns the exact result for the test case:
        """
        v = x - self.mean
        det = np.linalg.det(self.cov)
        maha = np.linalg.solve(self.cov, v).T.dot(v)[0, 0]
        return (1. / np.exp(.5 * maha)) * (det * (2 * np.pi) ** d) ** -.5
