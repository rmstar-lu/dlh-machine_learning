#!/usr/bin/env python3
"""
A function that updates a variable using the gradient descent with momentum optimization algorithm:

alpha is the learning rate
beta1 is the momentum weight
var is a numpy.ndarray containing the variable to be updated
grad is a numpy.ndarray containing the gradient of var
v is the previous first moment of var

Returns: the updated variable and the new moment, respectively
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Update variables using gradient descent with momentum optimization """
    v = beta1 * v - (1 - beta1) * alpha * grad
    return (var + v, v)
