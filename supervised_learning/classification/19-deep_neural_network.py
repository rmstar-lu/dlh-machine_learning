#!/usr/bin/env python3
"""
A class that defines a deep neural network performing binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        nx is the number of input features
        layers is a list representing the number of nodes per layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not (isinstance(layers, list) and layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev_dim = nx
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[f"W{i + 1}"] = \
                np.random.randn(layers[i], prev_dim) * np.sqrt(2 / prev_dim)
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
            prev_dim = layers[i]

    @property
    def L(self):
        """ getter for the L property """
        return self.__L

    @property
    def cache(self):
        """ getter for the cache property """
        return self.__cache

    @property
    def weights(self):
        """ getter for the weights property """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network
            X is of shape (nx, m) and contains the input data
        """
        def sigma(Z):
            """ activation function """
            return 1. / (1. + np.exp(-Z))

        A = self.__cache["A0"] = X
        for i in range(self.__L):
            A = self.__cache[f"A{i + 1}"] = \
                sigma(self.__weights[f"b{i + 1}"] +
                      self.__weights[f"W{i + 1}"] @ A)
        return (A, self.__cache)

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression (cross entropy)
            Y is of shape (1, m) and contains correct labels for the data
            A is of shape (1, m) and contains the activated output by example
        """
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
