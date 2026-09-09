#!/usr/bin/env python3
"""
A function that performs K-means on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """ A function that performs K-means on a dataset """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(k) is not int or k <= 0):
        return (None, None)

    n, d = X.shape
    centers = np.random.uniform(
        X.min(axis=0), X.max(axis=0), size=(k, d)
    )
    prev_centers = centers.copy()
    prev_centers.fill(np.nan)
    for i in range(iterations):
        dist = np.sqrt(((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
        labels = np.argmin(dist, axis=1)
        for j in range(k):
            if (labels == j).sum() == 0:
                centers[j] = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), size=(d,)
                )
            else:
                centers[j] = X[labels == j].mean(axis=0)
        if np.allclose(centers, prev_centers):
            return (centers, labels)
        prev_centers = centers.copy()
    return (None, None)
