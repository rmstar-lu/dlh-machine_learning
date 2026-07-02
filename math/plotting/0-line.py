#!/usr/bin/env python3
""" 0-line module: Plot y=x^3 as a line graph """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def line():
    """ Plot y=x^3 as a line graph. """

    x = np.arange(0, 11)
    y = x ** 3

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(x, y, color="red")

    ax.set(xlim=(0, 10))

    # plt.tight_layout()
    # plt.show()
