#!/usr/bin/env python3
"""
A function that converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix"""
    return 1. * (Y == np.arange(classes)[:, None])
