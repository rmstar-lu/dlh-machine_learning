#!/usr/bin/env python3
"""
A function that sets up Adam optimization for a keras model with categorical
crossentropy loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Set up Adam optimization with crossentropy loss and accuracy """
    adam = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )
