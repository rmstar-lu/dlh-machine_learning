#!/usr/bin/env python3
"""
Module binomial, defines a class representing a binomial distribution.
"""


class Binomial:
    """ Class representing a binomial distribution. """

    def __init__(self, data=None, n=1, p=0.5):
        """ constructor """

        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
        elif isinstance(data, list):
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            np = sum(data) / len(data)
            npq = sum((data_i - np) ** 2 for data_i in data) / len(data)
            q = npq / np
            n = round(np / (1 - q))
            p = np / n
        else:
            raise TypeError('data must be a list')
        self.n = int(n)
        self.p = float(p)
