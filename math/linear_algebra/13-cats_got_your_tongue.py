#!/usr/bin/env python3
""" 13-cats_got_your_tongue module, concatenate two numpy matrices. """

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy matrices along a specific axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
