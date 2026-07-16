#!/usr/bin/env python3
"""
Module normal, defines a class representing a normal distribution.
"""


class Normal:
    """ Class representing a normal distribution. """

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
