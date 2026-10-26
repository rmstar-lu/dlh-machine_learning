#!/usr/bin/env python3
"""
Functions to save and load an entire model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ Save an entire model """
    if filename[-6:] != ".keras":
        filename += ".keras"
    network.save(filename)


def load_model(filename):
    """ Load a model """
    return K.models.load_model(filename)
