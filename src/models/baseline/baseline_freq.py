import torch
import torch.nn as nn
from omegaconf import DictConfig
from utils.knowledge_db import TISSUES, CODON_MAP_DNA


class ModelBaseline(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device, pooling_dim=1024):
        super(ModelBaseline, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues

        layers = [
            nn.Linear(config.dim_embedding_tissue + len(CODON_MAP_DNA), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_size, 1)
        ]

        if config.binary_class:
            layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layers)

    @staticmethod
    def _compute_frequencies(rna_data):
        freqs = []
        for rna in rna_data:
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA)+1)[1:len(CODON_MAP_DNA)+1]
            freq = counts.float() / counts.sum()
            freqs.append(freq)
        return torch.stack(freqs)

    def forward(self, inputs):
        rna_data, tissue_id = inputs[0], inputs[1]  # (batch_size, padded_seq_length), upper bounded by max length

        rna_data_freq = self._compute_frequencies(rna_data)

        tissue_embedding = self.tissue_encoder(tissue_id)

        x = torch.cat((tissue_embedding, rna_data_freq), dim=1)

        y_pred = self.fc(x)

        return y_pred
