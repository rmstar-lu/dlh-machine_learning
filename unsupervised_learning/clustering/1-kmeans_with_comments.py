#!/usr/bin/env python3
"""
A function that performs K-means on a dataset:

X is a numpy.ndarray of shape (n, d) containing the dataset
n is the number of data points
d is the number of dimensions for each data point
k is a positive integer containing the number of clusters
iterations is a positive integer containing the maximum number of iterations
that should be performed
If no change in the cluster centroids occurs between iterations, your function
should return
Initialize the cluster centroids using a multivariate uniform distribution
(based on0-initialize.py)
If a cluster contains no data points during the update step, reinitialize
its centroid
You should use numpy.random.uniform exactly twice
You may use at most 2 loops
Returns: C, clss, or None, None on failure
C is a numpy.ndarray of shape (k, d) containing the centroid means for each
cluster
clss is a numpy.ndarray of shape (n,) containing the index of the cluster in
C that each data point belongs to
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
        # L2 distances to each center, dist.shape == (n, k)
        dist = np.sqrt(((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
        # find minimum index for each row
        labels = np.argmin(dist, axis=1)
        # update centers by averaging clusters with same label
        for j in range(k):
            if (labels == j).sum() == 0:
                # empty cluster, reinitialize centroid
                centers[j] = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), size=(d,)
                )
            else:
                centers[j] = X[labels == j].mean(axis=0)
        if np.allclose(centers, prev_centers):
            break
        prev_centers = centers.copy()
    # did not converge, reassign labels one last time
    dist = np.sqrt(((X[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    labels = np.argmin(dist, axis=1)
    return (centers, labels)
