#!/usr/bin/env python3
""" 3-flip_me_over module, transpose a 2D matrix. """


def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.
    This assumes the matrix is not empty and that all elements in each
    dimension are of the same type/shape.
    """
    return [list(row) for row in zip(*matrix)]
