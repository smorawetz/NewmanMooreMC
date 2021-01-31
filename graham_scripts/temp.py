import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# Stuff to make the plots look nice

params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 10,
    "figure.figsize": (10, 3),
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markeredgewidth": 0.8,
    "lines.markersize": 5,
    "lines.marker": "o",
    "patch.edgecolor": "black",
    "pgf.rcfonts": False,
}
plt.rcParams.update(params)
plt.style.use("seaborn-deep")


# Load in data and store in dictionary - NEW DATA


lattice_sizes = sys.argv[1].split(" ")
gamma = sys.argv[2]
stoptime = sys.argv[3]
final_temp = sys.argv[4]
seed = sys.argv[5]

sys.argv[1]
print(type(sys.argv[1]))
