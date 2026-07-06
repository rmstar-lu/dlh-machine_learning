#!/usr/bin/env python3
""" 2-change_scale module: Plot x->y as a line graph. """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def change_scale():
    """ Plot x->y as a line graph. """

    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.set(xlim=(0, 28650))
    ax.set_yscale('log', base=10)

    plt.title("Exponential Decay of C-14")
    plt.ylabel("Fraction Remaining")
    plt.xlabel("Time (years)")

    plt.plot(x, y, color="b")

    # plt.show()
