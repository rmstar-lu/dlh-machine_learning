#!/usr/bin/env python3
""" 4-line_up module, add arrays of the same shape element-wise. """


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.
    Return None if the shapes are different.
    """
    if len(arr1) != len(arr2):
        return None
    return [a1 + a2 for a1, a2 in zip(arr1, arr2)]
