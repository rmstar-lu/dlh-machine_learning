#!/usr/bin/env python3
"""
A function that creates a confusion matrix:

labels is a one-hot numpy.ndarray of shape (m, classes) containing the
correct labels for each data point
m is the number of data points
classes is the number of classes
logits is a one-hot numpy.ndarray of shape (m, classes) containing the
predicted labels

Returns: a confusion numpy.ndarray of shape (classes, classes) with row indices
representing the correct labels and column indices representing the
predicted labels
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Create a confusion matrix """
    m, classes = labels.shape
    conf = np.zeros((classes, classes))
    corr = np.nonzero(labels)[1]
    pred = np.argmax(logits, axis=1)
    np.add.at(conf, (corr, pred), 1)
    return conf
