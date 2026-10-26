#!/usr/bin/env python3
"""
A function to test a neural network.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Test a neural network """
    return network.evaluate(x=data, y=labels, verbose=2 if verbose else 0)
