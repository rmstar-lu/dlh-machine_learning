#!/usr/bin/env python3
"""
A function that updates the weights of a neural network with Dropout
regularization using gradient descent:

Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
correct labels for the data
    classes is the number of classes
    m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs and dropout masks of each layer of
the neural network
alpha is the learning rate
keep_prob is the probability that a node will be kept
L is the number of layers of the network

All layers use the tanh activation function except the last, which uses
the softmax activation function
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Update weights and biases using GD with dropout """

    def dTanh_dZ(Z):
        """ derivative of the tanh activation function """
        return 1 - Z ** 2

    classes, m = Y.shape
    dL_dZ_i_plus_1 = cache[f"A{L}"] - Y
    for i in range(L - 1, -1, -1):
        dL_dZ_i = ((weights[f"W{i + 1}"].T @ dL_dZ_i_plus_1)
                   * dTanh_dZ(cache[f"A{i}"]))
        dL_dZ_i *= cache.get(f"D{i}", 1) / keep_prob
        weights[f"W{i + 1}"] -= \
            alpha * ((dL_dZ_i_plus_1 / m) @ cache[f"A{i}"].T)
        weights[f"b{i + 1}"] -= \
            alpha * dL_dZ_i_plus_1.mean(axis=1, keepdims=True)
        dL_dZ_i_plus_1 = dL_dZ_i
