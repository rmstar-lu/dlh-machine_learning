#!/usr/bin/env python3
"""
A function that trains a model using mini-batch gradient descent.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """ Train the model using mini-batch gradient descent. """
    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=patience,
                             ))
        if learning_rate_decay:
            callbacks.append(K.callbacks.LearningRateScheduler(
                lambda epoch, lr: alpha / (1.0 + decay_rate * epoch),
                verbose=True,
            ))
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            save_best_only=True,
        ))
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2 if verbose else 0,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks,
    )
