#!/usr/bin/env python3
"""
A class that defines a neural network with one hidden layer
performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    A neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        nx is the number of input features to the neuron
        nodes is the number nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter for the W1 property """
        return self.__W1

    @property
    def b1(self):
        """ getter for the b1 property """
        return self.__b1

    @property
    def A1(self):
        """ getter for the A1 property """
        return self.__A1

    @property
    def W2(self):
        """ getter for the W2 property """
        return self.__W2

    @property
    def b2(self):
        """ getter for the b2 property """
        return self.__b2

    @property
    def A2(self):
        """ getter for the A2 property """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network
            X is of shape (nx, m) and contains the input data
        """
        def sigma(Z):
            """ activation function """
            return 1. / (1. + np.exp(-Z))

        self.__A1 = sigma(self.__b1 + self.__W1 @ X)
        self.__A2 = sigma(self.__b2 + self.__W2 @ self.__A1)
        return (self.__A1, self.__A2)
