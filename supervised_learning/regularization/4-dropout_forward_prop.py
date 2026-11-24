#!/usr/bin/env python3
"""
A function that conducts forward propagation using Dropout:

X is a numpy.ndarray of shape (nx, m) containing the input data for the network
    nx is the number of input features
    m is the number of data points
weights is a dictionary of the weights and biases of the neural network
L the number of layers in the network
keep_prob is the probability that a node will be kept
All layers except the last should use the tanh activation function
The last layer should use the softmax activation function

Returns: a dictionary containing the outputs of each layer and the dropout mask
used on each layer (see example for format)
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Conduct forward prop using dropout """

    def softmax(Z):
        """ softmax function for last layer """
        e = np.exp(Z)
        return e / e.sum(axis=0)

    d = {}
    A = d["A0"] = X
    for i in range(L - 1):
        D = d[f"D{i + 1}"] = 1 * (np.random.rand(weights[f"W{i + 1}"].shape[0],
                                                 A.shape[1]) < keep_prob)
        A = d[f"A{i + 1}"] = (D / keep_prob) * \
            np.tanh(weights[f"b{i + 1}"] + weights[f"W{i + 1}"] @ A)
    A = d[f"A{L}"] = softmax(weights[f"b{L}"] + weights[f"W{L}"] @ A)
    return d
