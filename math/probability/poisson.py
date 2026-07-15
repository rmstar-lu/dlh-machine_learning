#!/usr/bin/env python3
"""
Module poisson, defines a class representing a Poisson distribution.
"""


class Poisson:
    """ Class representing a Poisson distribution. """

    def __init__(self, data=None, lambtha=1.):
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
