#!/usr/bin/env python3
"""
7-high.py: A function that takes a pd.DataFrame as input and

Sorts it by the High price in descending order.
Returns: the sorted pd.DataFrame.
"""


def high(df):
    """ A function that sorts by descending High price. """

    return df.sort_values(by='High', ascending=False)
