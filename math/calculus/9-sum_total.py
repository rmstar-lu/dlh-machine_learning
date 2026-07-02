#!/usr/bin/env python3
""" 9-sum_total module: Calculate sum of i^2 """


def summation_i_squared(n):
    """ Calculate the sum of i^2 for i from 1 to n """

    if type(n) is not int:
        try:
            int_n = int(n)
        except (ValueError, TypeError):
            return None
        if int_n != n:
            return None
        n = int_n
    return sum(i ** 2 for i in range(1, n + 1))
