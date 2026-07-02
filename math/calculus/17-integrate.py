#!/usr/bin/env python3
""" 17-integrate module: Calculate the integral of a polynomial """


def poly_integral(poly, C=0):
    """ Calculate the integral of a polynomial. """

    if type(C) is not int or \
       not poly or not isinstance(poly, list) or \
       any(type(coeff) not in {int, float} or
           (type(coeff) is float and int(coeff) == coeff) for coeff in poly):
        return None
    integral = [C]
    for i, coeff in enumerate(poly):
        n = coeff / (i + 1)
        integral.append(int(n) if int(n) == n else n)
    # eliminate superfluous trailing 0's
    for i in range(len(integral) - 1, -1, -1):
        if integral[i] != 0:
            break
    return integral[:i+1]
