#!/usr/bin/env python3
"""
A function that creates mini-batches to be used for training a
neural network using mini-batch gradient descent.

X is a numpy.ndarray of shape (m, nx) representing input data
m is the number of data points
nx is the number of features in X
Y is a numpy.ndarray of shape (m, ny) representing the labels
m is the same number of data points as in X
ny is the number of classes for classification tasks.
batch_size is the number of data points in a batch
Returns: list of mini-batches containing tuples (X_batch, Y_batch)
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """ Create mini-batches to be used for training a neural network """
    X_s, Y_s = shuffle_data(X, Y)
    split = np.arange(batch_size, len(X), batch_size)
    return [(X_batch, Y_batch) for X_batch, Y_batch in
            zip(np.split(X_s, split), np.split(Y_s, split))]
