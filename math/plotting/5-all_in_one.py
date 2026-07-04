#!/usr/bin/env python3
""" 5-all_in_one module: Plot with subplots """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import matplotlib
# matplotlib.use('TKAgg')


def all_in_one():
    """ Plot with subplots. """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_axis_off()
    gs = GridSpec(3, 2, figure=fig)

    plt.title("All in One", fontsize="x-large")

    ax = fig.add_subplot(gs[0, 0])

    plt.plot(np.arange(0, 11), y0, color="red")
    ax.set(xlim=(0, 10))

    ax = fig.add_subplot(gs[0, 1])

    plt.title("Men's Height vs Weight", fontsize="small")
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    plt.scatter(x1, y1, marker='.', color="m")

    ax = fig.add_subplot(gs[1, 0])

    ax.set(xlim=(0, 28650))
    ax.set_yscale('log', base=10)

    plt.title("Exponential Decay of C-14", fontsize="small")
    plt.ylabel("Fraction Remaining", fontsize="small")
    plt.xlabel("Time (years)", fontsize="small")

    plt.plot(x2, y2, color="#598FC0", linewidth=1.5)

    ax = fig.add_subplot(gs[1, 1])

    ax.set(xlim=(0, 20000))
    ax.set(ylim=(0, 1))

    plt.title("Exponential Decay of Radioactive Elements", fontsize="small")
    plt.ylabel("Fraction Remaining", fontsize="small")
    plt.xlabel("Time (years)", fontsize="small")

    plt.plot(x3, y31, color="red", linestyle="--", label="C-14")
    plt.plot(x3, y32, color="green", label="Ra-226")

    ax.legend()

    ax = fig.add_subplot(gs[2, :])

    plt.title("Project A", fontsize="small")
    plt.ylabel("Number of Students", fontsize="small")
    plt.xlabel("Grades", fontsize="small")

    ax.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='k')

    ax.set(xlim=(0, 100))
    ax.set(ylim=(0, 30))

    plt.tight_layout()
    # plt.show()
