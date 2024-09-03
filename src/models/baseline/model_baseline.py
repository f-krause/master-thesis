import torch
import torch.nn as nn
from box import Box


class ModelBaseline(nn.Module):
    def __init__(self, config: Box, hidden_size=128):
        super(ModelBaseline, self).__init__()

        self.tissue_encoder = nn.Embedding(29, config.embedding_dim)

        # Define the layers
        self.input_size = config.embedding_dim + config.max_seq_length  # max seq length
        self.fc1 = nn.Linear(self.input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_size, 1)  # Hidden layer to output layer

    def forward(self, x):
        # FIXME generate data with padded input matrix
        # tissue_embedding = self.tissue_encoder(x["target_id"])
        # x = torch.nn.functional.pad(x["seq"], (0, 0, 0, self.input_size - x["seq"].shape[0]), value=-1)
        # x = torch.cat((tissue_embedding, x), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
