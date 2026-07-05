#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TKAgg')

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

plt.title("PCA of Iris Dataset", fontsize=25, y=1)
label_size=20
label_pad=15
tick_size=18
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_xlabel('U1', size=label_size, labelpad=label_pad)
ax.set_ylabel('U2', size=label_size, labelpad=label_pad)
ax.set_zlabel('U3', size=label_size, labelpad=label_pad)
ax.scatter3D(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], s=80, c=labels, cmap="plasma")
plt.tight_layout()
plt.show()
