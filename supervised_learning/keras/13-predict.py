#!/usr/bin/env python3
"""
A function to make a prediction using a neural network.
"""
import tensorflow.keras as K


def predict(network, data, verbose=True):
    """ Make a prediction """
    return network.predict(x=data, verbose=2 if verbose else 0)
