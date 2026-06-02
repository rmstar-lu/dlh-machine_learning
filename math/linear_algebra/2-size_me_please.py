#!/usr/bin/env python3
""" 2-size_me_please module, calculate the shape of a matrix. """


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.
    This assumes all elements in a dimension are of the same type/shape.
    """
    shape = []
    while type(matrix) is list:
        shape.append(len(matrix))
        if not matrix:
            break
        matrix = matrix[0]
    return shape
