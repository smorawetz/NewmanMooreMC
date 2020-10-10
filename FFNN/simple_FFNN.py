import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    def __init__(self, lattice_size, hidden_dim, output_dim):
        """
        lattice_size:       int
                            the size of the lattice
        hidden_size:        int
                            the number of hidden units in the FFNN
        output_size:        int
                            the number of output neurons
        """
        super(FFNN, self).__init__()

        self.input_dim = lattice_size ** 2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Sigmoid(),
        )

        def forward(lattice_config):
            return self.network(lattice_config)
