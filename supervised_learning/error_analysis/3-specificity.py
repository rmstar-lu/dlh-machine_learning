#!/usr/bin/env python3
"""
A function that calculates the specificity for each class in a
confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
    classes is the number of classes

Returns: a numpy.ndarray of shape (classes,) containing the specificity
of each class
"""
import numpy as np


def specificity(confusion):
    """ Calculate the specificity for each class in a confusion matrix """
    total_neg = confusion.sum() - np.sum(confusion, axis=1)
    true_neg = total_neg - np.sum(confusion, axis=0) + np.diag(confusion)
    return true_neg / total_neg
