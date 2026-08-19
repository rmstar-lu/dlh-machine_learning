#!/usr/bin/env python3
"""
8-prune.py: A function that takes a pd.DataFrame as input and

Removes any entries where Close has NaN values.
Returns: the modified pd.DataFrame.
"""


def prune(df):
    """ A function that filters non-NaN values from a DataFrame """

    return df[df.Close.notna()]
