#!/usr/bin/env python3
""" 8-ridin_bareback module, multiply two matrices. """


def mat_mul(mat1, mat2):
    """
    Multiply two matrices.
    """
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if len(mat1[0]) == 0 or len(mat2[0]) == 0:
        return None
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            for j in range(len(mat2[0]))] for i in range(len(mat1))]
