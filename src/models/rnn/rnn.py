import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from knowledge_db import TISSUES, CODON_MAP_DNA

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

        input_size = config.dim_embedding_tissue + config.dim_embedding_token

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

        # Repeat tissue embedding across the sequence length and concatenate to seq_embedding at each timestep
        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)  # (batch_size, padded_seq_length, dim_embedding_tissue)
        rnn_input = torch.cat((tissue_embedding_expanded, seq_embedding), dim=2)  # (batch_size, padded_seq_length, input_size)

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, seq_lengths.to('cpu'), batch_first=True,
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
