#!/usr/bin/env python3
"""
5-slice.py: A function that takes a pd.DataFrame as input and

Extracts the columns High, Low, Close, and Volume_(BTC).
Selects every 60th row from these columns.
Returns: the sliced pd.DataFrame
"""


def slice(df):
    """ A function that selects every 60th row from 4 columns """

    return df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]
