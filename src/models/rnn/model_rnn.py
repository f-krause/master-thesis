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

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=self.max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)  # 64 codons + padding 0

        # input_size = config.dim_embedding_tissue + config.dim_embedding_token
        input_size = config.dim_embedding_token

        if model.lower() == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                               bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)
        elif model.lower() == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=config.rnn_hidden_size, num_layers=config.num_layers,
                              bidirectional=config.bidirectional, dropout=config.dropout, batch_first=True)

        self.predictor = nn.Sequential(
            nn.Linear(config.num_layers * config.rnn_hidden_size, config.out_hidden_size),
            nn.ELU(),
            nn.Linear(config.out_hidden_size, 1),
        )

    def forward(self, inputs):
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data_pad = rna_data_pad.to(self.device)  # (batch_size, padded_seq_length), upper bounded by max length

        # Embedding for each component
        tissue_embedding = self.tissue_encoder(tissue_id)
        seq_embedding = self.seq_encoder(rna_data_pad)

        # Concatenate embeddings along the feature dimension
        # x = seq_embedding.flatten(start_dim=1)

        # x = torch.cat((tissue_embedding, x), dim=1)  # use?
        x = seq_embedding

        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True,
                                                           enforce_sorted=False)
        # How to include tissue embedding?
        h, c = self.rnn(x_packed)  # rnn(input, (h0, c0)), could also give tissue embedding as initial states
        h_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=0)

        y_pred = self.predictor(h_unpacked[:, -1, :])

        return y_pred
