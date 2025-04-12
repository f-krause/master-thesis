# GRU for "codon_mean_regression_train_8.1k_data.pkl" data, with targets being the mean across tissues

import torch
from torch import Tensor
import torch.nn as nn
from omegaconf import OmegaConf

from utils.knowledge_db import TISSUES, CODON_MAP_DNA


class ModelGRUMeanRegression(nn.Module):
    def __init__(self, config, device):
        super(ModelGRUMeanRegression, self).__init__()
        self.device = device
        # Use seq_encoder similar to other RNN models. No tissue embeddings.
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1,
                                        config.dim_embedding_token,
                                        padding_idx=0,
                                        max_norm=config.embedding_max_norm)
        self.gru = nn.GRU(
            input_size=config.dim_embedding_token,
            hidden_size=config.rnn_hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(config.rnn_hidden_size, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]
        # Packing the sequence

        x = self.seq_encoder(rna_data_pad)  # (batch, seq_length, dim_embedding_token)

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths.to('cpu'), batch_first=True,
                                                           enforce_sorted=False)

        out, _ = self.gru(x_packed)  # h_n: (1, batch, hidden_size)

        out_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=0)
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out_unpacked.size(2)))

        out_last = out_unpacked.gather(1, idx).squeeze(1)

        out_scalar = self.fc(out_last)  # (batch, 1)

        return out_scalar


if __name__ == "__main__":
    # Test model
    config_dev = OmegaConf.load("config/gru.yml")
    config_dev = OmegaConf.merge(config_dev,
                                 {"binary_class": True, "max_seq_length": 2700, "embedding_max_norm": 2})
    device_dev = torch.device("cpu")

    sample_batch = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (config_dev.batch_size, config_dev.max_seq_length)),
                                        batch_first=True),
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size, dtype=torch.int64)
        # seq_lengths (batch_size x 1)
    ]

    model = ModelGRUMeanRegression(config_dev, device_dev)

    print(model(sample_batch))