#!/usr/bin/env python3
""" 5-across_the_plances module, add 2D matrices of the same shape. """


def add_matrices2D(mat1, mat2):
    """
    Add two arrays element-wise.
    Return None if the shapes are different.
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1) == 0:
        return []
    if len(mat1[0]) != len(mat2[0]):
        return None

    return [[a1 + a2 for a1, a2 in zip(r1, r2)]
            for r1, r2 in zip(mat1, mat2)]
