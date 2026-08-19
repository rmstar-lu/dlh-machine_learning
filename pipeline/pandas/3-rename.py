#!/usr/bin/env python3
"""
3-rename.py: A function that takes a pd.DataFrame as input
and performs the following:

df is a pd.DataFrame containing a column named Timestamp.
The function should rename the Timestamp column to Datetime.
Convert the timestamp values to datetime values
Display only the Datetime and Close column
Returns: the modified pd.DataFrame
"""

import pandas as pd


def rename(df):
    """ A function that renames a column in a pd.DataFrame """

    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
