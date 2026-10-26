#!/usr/bin/env python3
"""
Functions to save and load a model's weights only.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """ Save a model's weights """
    ext = "." + save_format
    if filename[-len(ext):] != ext:
        filename += ext
    network.save_weights(filename)


def load_weights(network, filename):
    """ Load a model's weights """
    network.load_weights(filename)
