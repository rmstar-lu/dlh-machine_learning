#!/usr/bin/env python3
""" 7-gradient module: Plot contour as color gradient """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# import matplotlib
# matplotlib.use('TKAgg')


def gradient():
    """ Plot contour as color gradient """

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    plt.title("Mountain Elevation")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")

    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(z.min(), z.max())

    plt.scatter(x, y, marker=".", s=140, cmap='viridis', c=z)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, label="elevation (m)")
    # plt.gca().set_aspect('equal')
    # plt.show()
