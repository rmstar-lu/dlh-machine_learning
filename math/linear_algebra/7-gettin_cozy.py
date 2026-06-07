#!/usr/bin/env python3
""" 7-gettin_cozy module, concatenate two matrices. """


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.
    """
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    return [row1.copy() for row1 in mat1] + [row2.copy() for row2 in mat2]
