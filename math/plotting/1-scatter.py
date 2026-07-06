#!/usr/bin/env python3
""" 1-scatter module: Plot x->y as a scatter plot """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def scatter():
    """ Plot x->y as a scatter plot. """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Height (in)", fontsize=10)
    plt.ylabel("Weight (lbs)", fontsize=10)
    plt.title("Men's Height vs Weight")

    plt.scatter(x, y, c="m")

    # plt.show()
