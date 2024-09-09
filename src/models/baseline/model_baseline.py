import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box


class ModelBaseline(nn.Module):
    def __init__(self, config: Box, device: torch.device, hidden_size=128, pooling_dim=128):
        super(ModelBaseline, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length
        self.tissue_encoder = nn.Embedding(29, config.dim_embedding_tissue, padding_idx=0,
                                           max_norm=self.max_norm)  # 29 tissues in total
        self.seq_encoder = nn.Embedding(5, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)  # 4 nucleotides + padding
        self.sec_structure_encoder = nn.Embedding(4, config.dim_embedding_token, padding_idx=0,
                                                  max_norm=self.max_norm)  # 3 structures + padding
        self.loop_type_encoder = nn.Embedding(8, config.dim_embedding_token, padding_idx=0,
                                              max_norm=self.max_norm)  # 7 loop types + padding

        # Pooling to reduce the sequence dimension
        self.pooling_dim = pooling_dim
        self.pool = nn.AdaptiveAvgPool1d(self.pooling_dim)  # Reduce sequence length to pooling_dim

        # MLP layers
        self.fc1 = nn.Linear(config.dim_embedding_tissue + 3 * self.pooling_dim * config.dim_embedding_token,
                             hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        rna_data, tissue_id = zip(*x)
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data = torch.stack(rna_data).to(self.device)
        rna_data = rna_data.permute(0, 2, 1)

        # RNA data: (batch_size, 3, seq_length), where 3 is [seq, structure, loop]
        tissue_embedding = self.tissue_encoder(tissue_id)

        # Embedding for each component
        seq_embedding = self.seq_encoder(rna_data[:, 0])
        sec_structure_embedding = self.sec_structure_encoder(rna_data[:, 1])
        loop_type_embedding = self.loop_type_encoder(rna_data[:, 2])

        # Apply pooling to reduce sequence length
        seq_embedding = self.pool(seq_embedding.transpose(1, 2)).transpose(1, 2)
        sec_structure_embedding = self.pool(sec_structure_embedding.transpose(1, 2)).transpose(1, 2)
        loop_type_embedding = self.pool(loop_type_embedding.transpose(1, 2)).transpose(1, 2)

        # Concatenate embeddings along the feature dimension
        x = torch.cat((seq_embedding, sec_structure_embedding, loop_type_embedding), dim=2)
        x = x.flatten(start_dim=1)  # Flatten the pooled embeddings
        x = torch.cat((tissue_embedding, x), dim=1)

        # MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
