import torch
import torch.nn as nn
import torch.nn.functional as F
from box import Box
from knowledge_db import TISSUES, CODON_MAP_DNA


class ModelRNN(nn.Module):
    def __init__(self, config: Box, device: torch.device, model: str = "lstm"):
        super(ModelRNN, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.rnn_hidden_size,
                                           max_norm=self.max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)  # 64 codons + padding 0

        # input_size = config.dim_embedding_tissue + config.dim_embedding_token
        input_size = config.dim_embedding_token

        if model.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                               bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)
        elif model.lower() == "gru":
            # TODO TEST
            self.rnn = nn.GRU(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                              bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)

        self.predictor = nn.Sequential(
            # nn.LayerNorm(self.n_embd),
            nn.Linear((int(config.bidirectional) + 1) * config.rnn_hidden_size, config.out_hidden_size),
            nn.ELU(),
            nn.Linear(config.out_hidden_size, 1),
        )

    def forward(self, inputs):
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data_pad = rna_data_pad.to(self.device)  # (batch_size, padded_seq_length), upper bounded by max length

        # Embedding for each component
        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, tissue_embedding_dim)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, seq_embedding_dim)

        # Project tissue embedding to the LSTM hidden state dimension
        h0 = tissue_embedding.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)  # (num_layers, batch_size, hidden_dim)
        if self.rnn.bidirectional:
            h0 = h0.repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)  # (1, batch_size, hidden_dim)

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embedding, seq_lengths, batch_first=True,
                                                           enforce_sorted=False)

        if isinstance(self.rnn, nn.LSTM):
            h, _ = self.rnn(x_packed, (h0, c0))  # RNN forward pass with tissue embedding as initial states
        elif isinstance(self.rnn, nn.GRU):
            h, _ = self.rnn(x_packed, h0)
        else:
            raise ValueError("RNN type not supported")

        h_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=0)

        # Prediction based on last hidden state
        y_pred = self.predictor(h_unpacked[:, -1, :])

        return y_pred
