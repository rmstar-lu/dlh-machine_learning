#!/usr/bin/env python3
"""
A function that calculates the normalization (standardization) constants
of a matrix.
"""
import numpy as np


def normalization_constants(X):
    """ Calculate the standardization constants of a matrix """
    return (X.mean(axis=0), X.std(axis=0))
