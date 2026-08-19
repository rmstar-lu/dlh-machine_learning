#!/usr/bin/env python3
"""
6-flip_switch.py: A function that takes a pd.DataFrame as input and

Sorts the data in reverse chronological order.
Transposes the sorted dataframe.
Returns: the transformed pd.DataFrame.
"""


def flip_switch(df):
    """ A function that sorts data in reverse and transposes it. """

    return df.sort_values(by='Timestamp', ascending=False).T
