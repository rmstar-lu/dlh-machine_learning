#!/usr/bin/env python3
"""
A function that calculates the weighted moving average of a data set:

data is the list of data to calculate the moving average of
beta is the weight used for the moving average
Your moving average calculation should use bias correction
Returns: a list containing the moving averages of data
"""
import numpy as np


def moving_average(data, beta):
    """ Calculate a weighted moving average of the data """
    ema, r = 0, []
    for t, x in enumerate(data):
        ema = beta * ema + (1 - beta) * x
        r.append(ema / (1 - beta ** (1 + t)))
    return r
