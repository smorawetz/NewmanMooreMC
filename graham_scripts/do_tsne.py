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

data_dict = {}

for size in lattice_sizes:
    file_path = "stewart_data/L{0}/gamma{1}_stoptime{2}_T{3}_seed{4}.dat".format(
        size, gamma, stoptime, final_temp, seed
    )
    data = np.loadtxt(file_path)
    data_dict[size] = data

size_colors = {63: "red", 64: "green", 65: "blue"}
# size_colors = {61: "red", 62: "orange", 63: "yellow", 64: "green", 65: "blue", 66: "indigo", 67: "violet"}

embedded_data_dict = {}

perp = sys.argv[6]
lr = sys.argv[7]

for size in lattice_sizes:
    embedded_data = TSNE(n_components=2, perplexity=perp, learning_rate=lr).fit_transform(data_dict[size])
    embedded_data_dict[size] = embedded_data

fig, ax = plt.subplots(figsize=(9,5))

for size in lattice_sizes:
    plot_data = embedded_data_dict[size]
    ax.plot(plot_data[:,0],
            plot_data[:,1],
            "o",
            alpha=0.1 if size != 64 else 1,
            color=size_colors[size],
#             markersize=4,
            label="N = {0}".format(size)
           )

    
fig.legend(frameon=False)

plt.savefig("t-SNE_plots/t-SNE_gamma{0}_stoptime{1}_seed{2}_perp{3}_lr{4}.pdf".format(gamma, stoptime, seed, perp, lr), bbox_inches="tight")
