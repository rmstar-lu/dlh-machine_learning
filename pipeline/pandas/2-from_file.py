#!/usr/bin/env python3
"""
2-from_file.py: A function that loads data from a file as a pd.DataFrame
"""

import pandas as pd


def from_file(filename, delimiter):
    """ A function that loads data from a file as a pd.DataFrame """

    return pd.read_csv(filename, delimiter=delimiter)
