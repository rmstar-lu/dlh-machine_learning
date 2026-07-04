#!/usr/bin/env python3
""" 4-frequency module: Plot histogram """

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TKAgg')


def frequency():
    """ Plot histogram. """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    plt.title("Project A")
    plt.ylabel("Number of Students")
    plt.xlabel("Grades")

    ax.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='k')

    ax.set(xlim=(0, 100))
    ax.set(ylim=(0, 30))

    # plt.show()
