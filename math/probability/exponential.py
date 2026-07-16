#!/usr/bin/env python3
"""
Module exponential, defines a class representing an exponential distribution.
"""


class Exponential:
    """ Class representing an exponential distribution. """

    _E = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ constructor """

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        elif isinstance(data, list):
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            lambtha = len(data) / sum(data)
        else:
            raise TypeError('data must be a list')
        self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Probability density for a given time period.
        """

        if x < 0:
            return 0
        return self.lambtha * (self._E ** -self.lambtha)
