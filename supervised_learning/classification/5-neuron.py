#!/usr/bin/env python3
"""
A class that defines a single neuron performing binary classification.
"""
import numpy as np


class Neuron:
    """ A single neuron performing binary classification.  """

    def __init__(self, nx):
        """ nx is the number of input features to the neuron """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter for the W property """
        return self.__W

    @property
    def b(self):
        """getter for the b property """
        return self.__b

    @property
    def A(self):
        """ getter for the A property """
        return self.__A

    def forward_prop(self, X):
        """ X is of shape (nx, m) """
        Z = self.__b + self.__W @ X
        self.__A = 1. / (1. + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Cost using logistic regression (loss function)
            Y is of shape (1, m) and contains the correct labels
            A is of shape (1, m) containing the activated output by example
        """
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """ Return the neuron's predictions and the cost """
        A = self.forward_prop(X)
        return (A >= 0.5) * 1, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent, update weights and bias
            alpha is the learning rate
        """
        dL_dZ = A - Y  # derivative of loss by Z
        self.__W -= alpha * ((dL_dZ / X.shape[1]) @ X.T)
        self.__b -= alpha * dL_dZ.mean()
