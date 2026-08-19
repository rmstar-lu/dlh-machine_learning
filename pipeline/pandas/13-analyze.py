#!/usr/bin/env python3
"""
13-analyze.py: A function that takes a pd.DataFrame and:

Computes descriptive statistics for all columns except the Timestamp column.
Returns a new pd.DataFrame containing these statistics.
"""


def analyze(df):
    """ A function that computes descriptive statistics on a DataFrame """

    return df.drop(['Timestamp'], axis=1).describe()
