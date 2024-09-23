import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from knowledge_db import TISSUES, CODON_MAP_DNA


class ModelBaseline(nn.Module):
    def __init__(self, config: Box, device: torch.device, pooling_dim=1024):
        super(ModelBaseline, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=self.max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)  # 64 codons + padding 0

        # DOC POOLING
        # self.pooling_dim = pooling_dim
        # self.pool = nn.AdaptiveAvgPool1d(self.pooling_dim)  # Reduce sequence length to pooling_dim

        # self.fc1 = nn.Linear(config.dim_embedding_tissue + self.pooling_dim * config.dim_embedding_token,
        #                      hidden_size)

        # MLP layers
        self.fc1 = nn.Linear(config.dim_embedding_tissue + self.max_seq_length * config.dim_embedding_token,
                             config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.fc3 = nn.Linear(config.hidden_size // 2, 1)

    def forward(self, inputs):
        rna_data, tissue_id = inputs[0], inputs[1]
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data = rna_data.to(self.device) # (batch_size, padded_seq_length), upper bounded by max length
        # rna_data = torch.stack(rna_data).to(self.device)

        tissue_embedding = self.tissue_encoder(tissue_id)

        # padding to max_seq_length with 0
        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        # Embedding for each component
        tissue_embedding = self.tissue_encoder(tissue_id)
        seq_embedding = self.seq_encoder(rna_data_pad)

        # Apply pooling to reduce sequence length
        # x = self.pool(seq_embedding.transpose(1, 2)).transpose(1, 2)

        # Concatenate embeddings along the feature dimension
        seq_embedding_flat = seq_embedding.flatten(start_dim=1)
        x = torch.cat((tissue_embedding, seq_embedding_flat), dim=1)

        # MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
