#!/usr/bin/env python3
"""
A function that creates a batch normalization layer for a neural network
in tensorflow:

prev is the activated output of the previous layer
n is the number of nodes in the layer to be created
activation is the activation function that should be used on the output
of the layer

Returns: a tensor of the activated output for the layer.

I do not understand why this exercise asks to return the tensor instead of a
Layer object.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Create a batch normalization layer; return the output tensor """
    Z = tf.keras.layers.Dense(
        n,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        ),
    )(prev)

    # using the keras BatchNormalization layer returns different numbers, why?
    mu, variance = tf.nn.moments(Z, [0], keepdims=True)

    gamma = tf.Variable(tf.ones_like(mu), trainable=True)
    beta = tf.Variable(tf.zeros_like(mu), trainable=True)

    epsilon = 1e-7

    Z_norm = (Z - mu) / tf.sqrt(variance + epsilon)
    Z_norm = gamma * Z_norm + beta

    return activation(Z_norm)
