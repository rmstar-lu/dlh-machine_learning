#!/usr/bin/env python3
"""
A function that trains a model using mini-batch gradient descent:
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ Train the model using mini-batch gradient descent. """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2 if verbose else 0,
        shuffle=shuffle
    )
