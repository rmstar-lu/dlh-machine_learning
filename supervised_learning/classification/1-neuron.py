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
