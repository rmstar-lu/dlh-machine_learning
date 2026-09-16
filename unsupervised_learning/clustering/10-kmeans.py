#!/usr/bin/env python3
"""
A function that performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """ Perform k-means on a dataset using sklearn """
    KMeans = sklearn.cluster.KMeans(n_clusters=k)
    KMeans.fit(X)
    return (KMeans.cluster_centers_, KMeans.labels_)
