import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        # Algorithm from: https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html, see vaswani
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)  # TODO could include dropout here
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].to(x.device)
        return x  # self.dropout(x)


class ModelTransformer(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(ModelTransformer, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_token = config.dim_embedding_token
        self.dim_embedding_token = config.dim_embedding_token
        self.embedding_dim = self.dim_embedding_token

        # Embedding layers
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_token, max_norm=config.embedding_max_norm)
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

        self.predictor = Predictor(config, self.embedding_dim).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        # Embedding layers
        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        # Apply positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, embedding_dim)

        # Create attention mask (batch_size, seq_len)
        attention_mask = (rna_data_pad == 0)

        # Pass through Transformer Encoder: (seq_len, batch_size, embedding_dim)
        out = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)))  # (batch_size, 1, embedding_dim)
        out_last = out.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        y_pred = self.predictor(out_last)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    config_dev = OmegaConf.load("config/transformer.yml")
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

    model = ModelTransformer(config_dev, torch.device("cpu"))

    print(model(sample_batch))
