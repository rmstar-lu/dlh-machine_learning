#!/usr/bin/env python3
"""
A function that converts a one-hot matrix into a vector of labels
"""
import numpy as np


def one_hot_decode(one_hot):
    """Convert a one-hot matrix into a vector of labels"""
    if (not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2
            or one_hot.shape[0] < 1
            or not np.isin(one_hot, [0, 1]).all()):
        return None
    return np.nonzero(one_hot.T)[1]
