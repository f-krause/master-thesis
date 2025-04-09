import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor


class ModelRNN(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device, model: str = "lstm"):
        super(ModelRNN, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0

        input_size = config.dim_embedding_token

        if model.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                               bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)
        elif model.lower() == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                              bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)
        else:
            raise ValueError(f"Model {model} not supported: either 'lstm' or 'gru'")

        self.predictor = Predictor(config, (int(config.bidirectional) + 1) * config.rnn_hidden_size).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths.to('cpu'), batch_first=True,
                                                           enforce_sorted=False)

        if isinstance(self.rnn, nn.LSTM):
            h, _ = self.rnn(x_packed)
        elif isinstance(self.rnn, nn.GRU):
            h, _ = self.rnn(x_packed)
        else:
            raise ValueError(f"RNN type {type(self.rnn)} not supported")

        h_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=0)

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h_unpacked.size(2)))
        h_last = h_unpacked.gather(1, idx).squeeze(1)

        y_pred = self.predictor(h_last)

        return y_pred


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

    model = ModelRNN(config_dev, device_dev)

    print(model(sample_batch))
