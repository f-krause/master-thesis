import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box


class ModelBaseline(nn.Module):
    def __init__(self, config: Box, device: torch.device, hidden_size=256, reduced_dim=512):
        super(ModelBaseline, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.tissue_encoder = nn.Embedding(29, config.dim_embedding_tissue, padding_idx=0)  # 29 tissues in total
        self.seq_encoder = nn.Embedding(5, config.dim_embedding_token, padding_idx=0)  # 4 nucleotides in total + padding
        self.sec_structure_encoder = nn.Embedding(4, config.dim_embedding_token, padding_idx=0)  # 3 secondary structures in total + padding
        self.loop_type_encoder = nn.Embedding(8, config.dim_embedding_token, padding_idx=0)  # 7 loop types in total + padding

        # Define the layers
        # self.input_size = config.dim_embedding_tissue + config.max_seq_length * 3 * config.dim_embedding_token  # max seq length
        # TODO add pooling here
        self.fc1 = nn.Linear(self.reduced_dim, hidden_size*2)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_size, 1)  # Hidden layer to output layer

    def forward(self, x):
        rna_data, tissue_id = zip(*x)
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data = torch.stack(rna_data).to(self.device)

        # FIXME padding rna_data to max_seq_length
        # torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

        rna_data = rna_data.permute(0, 2, 1)

        tissue_embedding = self.tissue_encoder(tissue_id.to(self.device))

        seq_embedding = self.seq_encoder(rna_data[:, 0].to(self.device))
        sec_structure_embedding = self.sec_structure_encoder(rna_data[:, 1].to(self.device))
        loop_type_embedding = self.loop_type_encoder(rna_data[:, 2].to(self.device))

        # pad embeddings to max_seq_length
        padding_size = self.max_seq_length - rna_data.shape[2]
        seq_embedding = F.pad(seq_embedding, (0, 0, 0, padding_size), mode='constant', value=0)
        sec_structure_embedding = F.pad(sec_structure_embedding, (0, 0, 0, padding_size), mode='constant', value=0)
        loop_type_embedding = F.pad(loop_type_embedding, (0, 0, 0, padding_size), mode='constant', value=0)

        # Concatenate the embeddings
        x = torch.cat((seq_embedding, sec_structure_embedding, loop_type_embedding), dim=1)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((tissue_embedding, x), dim=1)

        # TODO embed to lower space? To large MLP!


        # Actual MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
