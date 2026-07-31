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
        def cov(x, y):
            """ covariance """
            return ((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1)

        if not (isinstance(data, np.ndarray) and len(data.shape) == 2):
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1).reshape((d, 1))
        self.cov = np.array([[cov(data[i], data[j]) for i in range(d)]
                            for j in range(d)])

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

        v = x - self.mean
        return ((2 * np.pi) ** (-.5 * d) * np.linalg.det(self.cov) ** -.5 *
                np.exp(-.5 * (v.T @ np.linalg.inv(self.cov) @ v)[0, 0]))
