#!/usr/bin/env python3
"""
A function that creates a neural network layer in tensorFlow that includes
L2 regularization:

prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function that should be used on the layer
lambtha is the L2 regularization parameter

Returns: the output of the new layer
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Create a L2 regularization layer; return the output tensor """
    return tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode=("fan_avg"),
        ),
        kernel_regularizer=tf.keras.regularizers.L2(l2=lambtha),
    )(prev)
