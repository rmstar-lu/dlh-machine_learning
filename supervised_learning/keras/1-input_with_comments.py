#!/usr/bin/env python3
"""
A function that builds a neural network with the Keras library.

nx is the number of input features to the network
layers is a list containing the number of nodes in each layer of the network
activations is a list containing the activation functions used for each layer of the network
lambtha is the L2 regularization parameter
keep_prob is the probability that a node will be kept for dropout

Returns: the keras model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Build a neural network using the functional API """
    x = inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1-keep_prob)(x)
    return K.Model(inputs=inputs, outputs=x)
