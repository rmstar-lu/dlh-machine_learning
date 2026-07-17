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

    def pmf(self, k):
        """ Probability for x=k """

        k = int(k)
        if k < 0:
            return 0
        n_choose_k = 1
        for i in range(k + 1, self.n + 1):
            n_choose_k *= i
        for i in range(2, self.n - k + 1):
            n_choose_k //= i
        return n_choose_k * self.p ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        """ Probability for x <= k """

        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(j) for j in range(k + 1))
