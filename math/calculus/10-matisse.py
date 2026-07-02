#!/usr/bin/env python3
""" 10-matisse module: Calculate the derivative of a polynomial """


def poly_derivative(poly):
    """ Calculate the derivative of a polynomial. """

    if not poly or not isinstance(poly, list) or \
       any(type(coeff) not in {int, float} for coeff in poly):
        return None
    # eliminate superfluous trailing 0's
    for i in range(len(poly) - 1, -1, -1):
        if poly[i] != 0:
            break
    poly = poly[:i+1]
    deriv = []
    for i, coeff in enumerate(poly):
        if i > 0:
            deriv.append(coeff * i)
    return deriv if deriv else [0]
