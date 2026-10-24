#!/usr/bin/env python3
"""
A function that builds a neural network with the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Build a neural network. """
    model = K.Sequential()
    model.add(K.Input(shape=(nx,)))
    l2 = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=l2
        ))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1-keep_prob))
    return model
