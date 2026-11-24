#!/usr/bin/env python3
"""
A function that updates the weights and biases of a neural network using
gradient descent with L2 regularization.

Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct
labels for the data
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs of each layer of the neural network
alpha is the learning rate
lambtha is the L2 regularization parameter
L is the number of layers of the network
The neural network uses tanh activations on each layer except the last,
which uses a softmax activation
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Update weights and biases using GD with L2 regularization """

    def dTanh_dZ(Z):
        """ derivative of the tanh activation function """
        return 1 - Z ** 2

    classes, m = Y.shape
    dL_dZ_i_plus_1 = cache[f"A{L}"] - Y
    for i in range(L - 1, -1, -1):
        dL_dZ_i = ((weights[f"W{i + 1}"].T @ dL_dZ_i_plus_1)
                   * dTanh_dZ(cache[f"A{i}"]))
        weights[f"W{i + 1}"] -= \
            alpha * ((dL_dZ_i_plus_1 / m) @ cache[f"A{i}"].T +
                     (lambtha / m) * weights[f"W{i+1}"])
        weights[f"b{i + 1}"] -= \
            alpha * dL_dZ_i_plus_1.mean(axis=1, keepdims=True)
        dL_dZ_i_plus_1 = dL_dZ_i
