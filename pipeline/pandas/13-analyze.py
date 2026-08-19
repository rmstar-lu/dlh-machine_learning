#!/usr/bin/env python3
"""
13-analyze.py: A function that takes a pd.DataFrame and:

Computes descriptive statistics for all columns except the Timestamp column.
Returns a new pd.DataFrame containing these statistics.
"""

import pandas as pd


def analyze(df):
    """ A function that computes descriptive statistics on a DataFrame """

    df = df.drop(['Timestamp'], axis=1)
    d = dict()
    for k in ['count', 'mean', 'std', 'min', 'max']:
        d[k] = getattr(df, k)()
    return pd.DataFrame(d).T
