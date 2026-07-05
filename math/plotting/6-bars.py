#!/usr/bin/env python3
""" 6-bars module: Stacked bar plot """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def bars():
    """ Stacked bar plot. """

    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    persons = ["Farrah", "Fred", "Felicia"]
    kinds = ["apples", "bananas", "oranges", "peaches"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")

    bottom = np.zeros(len(persons))
    for k, c, f in zip(kinds, colors, fruit):
        ax.bar(persons, f, 0.5, label=k, color=c, bottom=bottom)
        bottom += f

    ax.set(ylim=(0, 80))
    ax.legend(loc="upper right")

    # plt.show()
