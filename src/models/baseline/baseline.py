import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from knowledge_db import TISSUES, CODON_MAP_DNA


class ModelBaseline(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device, pooling_dim=1024):
        super(ModelBaseline, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0

        layers = [
            nn.Linear(config.dim_embedding_tissue + self.max_seq_length * config.dim_embedding_token,
                      config.hidden_size),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        ]

        if config.binary_class:
            layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layers)

        # DOC POOLING
        # self.pooling_dim = pooling_dim
        # self.pool = nn.AdaptiveAvgPool1d(self.pooling_dim)  # Reduce sequence length to pooling_dim

        # first layer of MLP: input size
        # nn.Linear(config.dim_embedding_tissue + self.pooling_dim * config.dim_embedding_token,
        #                      hidden_size)

    def forward(self, inputs):
        rna_data, tissue_id = inputs[0], inputs[1]  # (batch_size, padded_seq_length), upper bounded by max length

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        tissue_embedding = self.tissue_encoder(tissue_id)
        seq_embedding = self.seq_encoder(rna_data_pad)
        seq_embedding_flat = seq_embedding.flatten(start_dim=1)

        # Apply pooling to reduce sequence length
        # x = self.pool(seq_embedding.transpose(1, 2)).transpose(1, 2)

        x = torch.cat((tissue_embedding, seq_embedding_flat), dim=1)

        y_pred = self.fc(x)

        return y_pred
