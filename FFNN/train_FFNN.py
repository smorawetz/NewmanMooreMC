import numpy as np
import torch

import torch.optim as optim

from simple_FFNN import FFNN  # locally defined network structure


# Define physical parameters
lattice_size = 64

# Define NN parameters
hidden_dim = 100
output_dim = 2

# Define training parameters
BATCH_SIZE = 50
PERIOD = 10
SEED = 1


# Do training


def fit(lattice_size, hidden_dim, output_dim, lr, train_epochs):
    """
    lattice_size:   int
                    the size of the lattice
    hidden_dim:     int
                    the number of hidden neurons
    output_dim:     int
                    the number of output neurons
    lr:             float
                    the learning rate
    train_epochs:   int
                    the number of epochs of training
    """
    torch.manual_seed(SEED)

    # Find and load in the training data
    data_path = "../data_Aug20/L{0}/T0.406569.dat.gz".format(lattice_size)
    training_data = torch.Tensor(np.loadtxt(data_path))

    # Define folder to store training results in
    results_path = "training_results/L{0}_nh{1}_nout{2}_lr{3}".format(
        lattice_size, hidden_dim, output_dim, lr
    )

    # Instantiate NN model

    model = FFNN(lattice_size, hidden_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)
