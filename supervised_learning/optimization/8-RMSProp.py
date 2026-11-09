#!/usr/bin/env python3
"""
A function that sets up the RMSProp optimization algorithm
in tensorflow:

alpha is the learning rate
beta2 is the RMSProp weight (Discounting factor)
epsilon is a small number to avoid division by zero

Returns: optimizer
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ Create a RMSProp optimizer """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon,
    )
