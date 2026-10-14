#!/usr/bin/env python3
"""
A class that defines a deep neural network performing
multiclass classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    A deep neural network performing multiclass classification.
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

        def softmax(Z):
            """ softmax function for last layer """
            e = np.exp(Z)
            return e / e.sum(axis=0)

        A = self.__cache["A0"] = X
        for i in range(self.__L - 1):
            A = self.__cache[f"A{i + 1}"] = \
                sigma(self.__weights[f"b{i + 1}"] +
                      self.__weights[f"W{i + 1}"] @ A)
        A = self.__cache[f"A{self.__L}"] = \
            softmax(self.__weights[f"b{self.__L}"] +
                    self.__weights[f"W{self.__L}"] @ A)
        return (A, self.__cache)

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression (cross entropy)
            Y is of shape (classes, m) and contains one-hot encoded labels
            A is of shape (classes, m) and contains the softmax values
        """
        return np.mean(-np.sum(Y * np.log(A), axis=0))

    def evaluate(self, X, Y):
        """ Return the neural network's prediction and the cost """
        A, _ = self.forward_prop(X)
        one_hot = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return (one_hot, self.cost(Y, A))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculate one pass of gradient descent, update weights and biases
            alpha is the learning rate
        """
        nx, m = cache["A0"].shape
        dL_dZ_i_plus_1 = cache[f"A{self.__L}"] - Y
        for i in range(self.__L - 1, -1, -1):
            dL_dZ_i = ((self.__weights[f"W{i + 1}"].T @ dL_dZ_i_plus_1)
                       * (cache[f"A{i}"] * (1 - cache[f"A{i}"])))
            self.__weights[f"W{i + 1}"] -= \
                alpha * ((dL_dZ_i_plus_1 / m) @ cache[f"A{i}"].T)
            self.__weights[f"b{i + 1}"] -= \
                alpha * dL_dZ_i_plus_1.mean(axis=1, keepdims=True)
            dL_dZ_i_plus_1 = dL_dZ_i

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Train the neural network """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        graph_x = []
        graph_y = []
        for t in range(iterations):
            A, cache = self.forward_prop(X)
            if (verbose or graph) and t % step == 0:
                cost = self.cost(Y, A)
                if verbose:
                    print(f"Cost after {t} iterations: {cost}")
                if graph:
                    graph_x.append(t)
                    graph_y.append(cost)
            self.gradient_descent(Y, cache, alpha)
        predictions, cost = self.evaluate(X, Y)
        if verbose:
            print(f"Cost after {iterations} iterations: {cost}")
        if graph:
            graph_x.append(iterations)
            graph_y.append(cost)
            fig = plt.figure(figsize=(5.4, 4.4))
            plt.plot(graph_x, graph_y)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.tight_layout()
            plt.show()
        return (predictions, cost)

    def save(self, filename):
        """Save the network to a file in pickle format"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled network from a file"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except OSError:
            return None
