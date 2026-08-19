#!/usr/bin/env python3
"""
4-array.py: A function that takes a pd.DataFrame as input
and performs the following:

df is a pd.DataFrame containing columns named High and Close.
The function should select the last 10 rows of the High and Close columns.
Convert these selected values into a numpy.ndarray.
Returns: the numpy.ndarray
"""

import pandas as pd


def array(df):
    """ A function that selects the last 10 of columns High and Close. """

    return df[['High', 'Close']].tail(10).values
