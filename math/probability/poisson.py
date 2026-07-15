#!/usr/bin/env python3
"""
Module poisson, defines a class representing a Poisson distribution.
"""


class Poisson:
    """ Class representing a Poisson distribution. """

    _E = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ constructor """

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        elif isinstance(data, list):
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            lambtha = sum(data) / len(data)
        else:
            raise TypeError('data must be a list')
        self.lambtha = float(lambtha)

    def pmf(self, k):
        """
        Probability for x=k.
        This will overflow for large k, but I am not allowed to use ln().
        Normally I would divide by successive i instead of multiplying out
        the factorial.
        """

        if k < 0:
            return 0
        k_fac = 1
        for i in range(2, k + 1):
            k_fac *= i
        return ((self.lambtha ** k) * (self._E ** -self.lambtha)) / k_fac
