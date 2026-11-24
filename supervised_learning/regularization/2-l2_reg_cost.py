#!/usr/bin/env python3
"""
A function that calculates the cost of a neural network with L2 regularization:

cost is a tensor containing the cost of the network without L2 regularization
model is a Keras model that includes layers with L2 regularization

Returns: a tensor containing the total cost for each layer of the network,
accounting for L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """ Calculate the cost of a neural network with L2 regularization """
    l2_cost = []
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer") and \
                layer.kernel_regularizer is not None:
            l2_cost.append(layer.kernel_regularizer.l2 *
                           tf.reduce_sum(tf.square(layer.weights[0])))
    return tf.add(cost, tf.convert_to_tensor(l2_cost, dtype='float32'))
