import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA


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
            nn.Linear(self.max_seq_length * config.dim_embedding_token, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
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

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        x = x.flatten(start_dim=1)

        # Apply pooling to reduce sequence length (very old legacy)
        # x = self.pool(seq_embedding.transpose(1, 2)).transpose(1, 2)

        y_pred = self.fc(x)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    config_dev = OmegaConf.load("config/baseline.yml")
    config_dev = OmegaConf.merge(config_dev,
                                 {"binary_class": True, "max_seq_length": 2700, "embedding_max_norm": 2})

    sample_batch = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (config_dev.batch_size, config_dev.max_seq_length)),
                                        batch_first=True),
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size, dtype=torch.int64)
        # seq_lengths (batch_size x 1)
    ]

    model = ModelBaseline(config_dev, torch.device("cpu"))

    print(model(sample_batch))
