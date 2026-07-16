#!/usr/bin/env python3
"""
Module normal, defines a class representing a normal distribution.
"""


class Normal:
    """ Class representing a normal distribution. """

    _PI = 3.1415926536
    _E = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """ constructor """

        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        elif isinstance(data, list):
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((data_i - mean) ** 2 for data_i in data) / len(data)
            stddev = variance ** .5
        else:
            raise TypeError('data must be a list')
        self.mean = float(mean)
        self.stddev = float(stddev)

    def z_score(self, x):
        """ z score of given x value """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ x value of given z-score """

        return self.mean + z * self.stddev

    def pdf(self, x):
        """ Probability density for given x-value """

        return ((self._E ** -((x - self.mean) ** 2 / (2 * self.stddev ** 2))) /
                (self.stddev * (2 * self._PI) ** .5))
