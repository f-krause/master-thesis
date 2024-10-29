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
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_token = config.dim_embedding_token
        self.tissue_embedding_dim = config.tissue_embedding_dim
        self.embedding_dim = self.dim_embedding_token + len(CODON_MAP_DNA)

        # Embedding layers
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.tissue_embedding_dim, max_norm=config.embedding_max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, self.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=self.max_seq_length)

        # Transformer Encoder
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,  # input size
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
            norm=nn.LayerNorm(self.embedding_dim),
        )

        self.predictor = Predictor(self.embedding_dim, config.out_hidden_size).to(self.device)

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

        # Embedding layers
        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, tissue_embedding_dim)
        # seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)
        seq_embedding = self._compute_frequencies(rna_data_pad)

        # Expand tissue embedding to match sequence length
        # tissue_embedding_expanded = tissue_embedding.unsqueeze(1).repeat(1, seq_embedding.size(1), 1)  # (batch_size, seq_len, tissue_embedding_dim)

        # Concatenate sequence embedding and tissue embedding
        combined_embedding = torch.cat((seq_embedding, tissue_embedding), dim=1).unsqueeze(1)  # (batch_size, seq_len, embedding_dim)

        # Apply positional encoding
        combined_embedding = self.positional_encoding(combined_embedding)  # (batch_size, seq_len, embedding_dim)

        # Create attention mask (batch_size, seq_len)
        attention_mask = (seq_embedding == 0)

        # Pass through Transformer Encoder: (seq_len, batch_size, embedding_dim)
        out = self.transformer_encoder(combined_embedding)#, src_key_padding_mask=attention_mask)

        # Extract outputs corresponding to the last valid time step
        # idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)))  # (batch_size, 1, embedding_dim)
        # out_last = out.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        y_pred = self.predictor(out)

        return y_pred
