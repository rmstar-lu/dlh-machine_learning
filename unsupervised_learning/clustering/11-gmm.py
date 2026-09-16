#!/usr/bin/env python3
"""
A function that calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """ Calculate a GMM from a dataset using sklearn """
    GMM = sklearn.mixture.GaussianMixture(k)
    GMM.fit(X)
    return (
        GMM.weights_, GMM.means_, GMM.covariances_, GMM.predict(X), GMM.bic(X)
    )
