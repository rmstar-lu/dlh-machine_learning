#!/usr/bin/env python3
"""
A function that sets up the gradient descent with momentum optimization
in tensorflow:

alpha is the learning rate
beta1 is the momentum weight

Returns: optimizer
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """ Create a momentum optimizer """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1,
    )
