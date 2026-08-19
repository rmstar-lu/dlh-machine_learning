#!/usr/bin/env python3
"""
9-fill.py: A function that takes a pd.DataFrame as input and

Removes the Weighted_Price column.
Fills missing values in the Close column with the previous row’s value.
Fills missing values in the High, Low, and Open columns with the
corresponding Close value in the same row.
Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
Returns: the modified pd.DataFrame.
"""


def fill(df):
    """ A function that fills missing values in a DataFrame """

    df = df.drop(['Weighted_Price'], axis=1)
    df.Close = df.Close.ffill()
    df.High = df.High.fillna(df.Close)
    df.Low = df.Low.fillna(df.Close)
    df.Open = df.Open.fillna(df.Close)
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
