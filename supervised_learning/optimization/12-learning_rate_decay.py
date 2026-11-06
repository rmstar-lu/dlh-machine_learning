#!/usr/bin/env python3
"""
A function that creates an inverse time decay learning rate decay operation
in tensorflow:

alpha is the original learning rate
decay_rate is the weight used to determine the rate at which alpha will decay
decay_step is the number of passes of gradient descent that should occur
before alpha is decayed further
the learning rate decay should occur in a stepwise fashion

Returns: the learning rate decay operation
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """ Create a learning rate decay operation in tensorflow """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha, decay_step, decay_rate, staircase=True
    )
