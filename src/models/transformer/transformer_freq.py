import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # TODO figure out what is going on here
        # https://medium.com/image-processing-with-python/positional-encoding-in-the-transformer-model-e8e9979df57f
        # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        # Algorithm from: https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html, se vaswani

        """
        Positional Encoding Module to apply positional encodings up to the given max_len.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings up to max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer("pe", pe)  # Register as a buffer to persist during model operations

    def forward(self, x):
        # Dynamically select positional encodings based on input sequence length
        seq_len = x.size(1)  # Get the sequence length of the input
        x = x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)
        return x


class ModelTransformer(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(ModelTransformer, self).__init__()

        self.device = device
        self.input_dim = config.dim_embedding_tissue + len(CODON_MAP_DNA)
        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue, max_norm=config.embedding_max_norm)
        self.positional_encoding = PositionalEncoding(self.input_dim, max_len=config.max_seq_length)

        # Transformer Encoder
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,  # input size
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(self.input_dim),
        )

        self.predictor = Predictor(config, self.input_dim).to(self.device)

    @staticmethod
    def _compute_frequencies(rna_data):
        freqs = []
        for rna in rna_data:
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA)+1)[1:len(CODON_MAP_DNA)+1]
            freq = counts.float() / counts.sum()
            freqs.append(freq)
        return torch.stack(freqs)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, tissue_embedding_dim)
        seq_embedding = self._compute_frequencies(rna_data_pad)
        combined_embedding = torch.cat((seq_embedding, tissue_embedding), dim=1).unsqueeze(1)

        combined_embedding = self.positional_encoding(combined_embedding)  # (batch_size, seq_len, embedding_dim)

        out = self.transformer_encoder(combined_embedding)

        y_pred = self.predictor(out)

        return y_pred
