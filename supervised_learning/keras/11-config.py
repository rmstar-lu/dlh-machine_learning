#!/usr/bin/env python3
"""
Functions to save and load a model's configuration in JSON format.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """ Save a model's configuration in JSON format """
    if filename[-5:] != ".json":
        filename += ".json"
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """ Load a model with a saved configuration in JSON format """
    with open(filename, "r") as f:
        return K.models.model_from_json(f.read())
