#!/usr/bin/env python3
"""
10-index.py: A function that takes a pd.DataFrame as input and

Sets the Timestamp column as the index of the dataframe.
Returns: the modified pd.DataFrame.
"""


def index(df):
    """ A function that indexes a DataFrame on Timestamp """

    return df.set_index('Timestamp')
