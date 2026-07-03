#!/usr/bin/env python3
""" 3-two module: Plot x->y1 and x->y2 as line graphs. """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def two():
    """ Plot x->y1 and x->y2 as line graphs. """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.set(xlim=(0, 20000))
    ax.set(ylim=(0, 1))

    plt.title("Exponential Decay of Radioactive Elements")
    plt.ylabel("Fraction Remaining", fontsize=10)
    plt.xlabel("Time (years)", fontsize=10)

    plt.plot(x, y1, color="red", linestyle="--", label="C-14")
    RA_226 = plt.plot(x, y2, color="green", label="Ra-226")

    ax.legend()
    # plt.show()
