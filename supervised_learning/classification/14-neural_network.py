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
        self.__b2 = np.zeros((1, 1))
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

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression (cross entropy)
            Y is of shape (1, m) and contains correct labels for the data
            A is of shape (1, m) and contains the activated output by example
        """
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """ Return the neural network's prediction and the cost """
        _, A = self.forward_prop(X)
        return (A >= 0.5) * 1, self.cost(Y, A)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculate one pass of gradient descent, update weights and biases
            alpha is the learning rate
        """
        nx, m = X.shape
        dL_dZ2 = A2 - Y  # derivative of loss by Z2
        dL_dZ1 = (self.__W2.T @ dL_dZ2) * (A1 * (1 - A1))  # shape (nodes, m)
        self.__W1 -= alpha * ((dL_dZ1 / m) @ X.T)
        self.__b1 -= alpha * dL_dZ1.mean(axis=1, keepdims=True)
        self.__W2 -= alpha * ((dL_dZ2 / m) @ A1.T)
        self.__b2 -= alpha * dL_dZ2.mean()

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train the neural network """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for t in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
