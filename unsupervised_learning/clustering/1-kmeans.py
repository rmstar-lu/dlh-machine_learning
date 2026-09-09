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
from sklearn.cluster import KMeans


def kmeans(X, k, iterations=1000):
    """ A function that performs K-means on a dataset """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2
            or X.shape[0] < 1 or X.shape[1] < 1
            or type(k) is not int or k <= 0):
        return (None, None)

    d = X.shape[1]
    centers = np.random.uniform(
        X.min(axis=0), X.max(axis=0), size=(k, d)
    )
    reinitialized = True
    while reinitialized:
        kmeans = KMeans(n_clusters=k, init=centers, max_iter=iterations)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        reinitialized = False
        for i in range(d):
            if (kmeans.labels_ == i).sum() == 0:
                # reinitialize center of empty cluster
                centers[i] = np.random.uniform(
                    X.min(axis=0), X.max(axis=0), size=(d,)
                )
                reinitialized = True
    return (centers, kmeans.labels_)
