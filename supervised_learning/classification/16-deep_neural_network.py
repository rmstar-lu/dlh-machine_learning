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
        if not (isinstance(layers, list) and list
                and all(type(i) is int and i >= 1 for i in layers)):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev_dim = nx
        for i in range(self.L):
            self.weights[f"W{i + 1}"] = \
                np.random.randn(layers[i], prev_dim) * np.sqrt(2 / prev_dim)
            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
            prev_dim = layers[i]
        self.A2 = 0
