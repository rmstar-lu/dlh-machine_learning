#!/usr/bin/env python3
""" 9-sum_total module: Calculate sum of i^2 """


def summation_i_squared(n):
    """ Calculate the sum of i^2 for i from 1 to n """

    if type(n) is not int:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
