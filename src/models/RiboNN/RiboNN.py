import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from omegaconf import OmegaConf, DictConfig

from utils.knowledge_db import TISSUES, CODON_MAP_DNA, TOKENS
from models.predictor import Predictor


class ConvBlockRiboNN(nn.Module):
    """
    One conv-block:
      1) LayerNorm over channels (via transpose)
      2) ReLU
      3) Conv1d(kernel_size=5)
      4) Dropout
      5) MaxPool1d(kernel_size=2)
    """
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, stride=1)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        # apply LayerNorm on C → need to swap (B,L,C)
        x = x.transpose(1, 2)       # → (B, L, C)
        x = self.norm(x)
        x = x.transpose(1, 2)       # → (B, C, L)
        x = self.act(x)
        x = self.conv(x)            # → (B, C, L-4)
        x = self.drop(x)
        x = self.pool(x)            # → (B, C, ⌊(L-4)/2⌋)
        return x


class RiboNN(nn.Module):
    """
    Conv-based drop-in replacement for the original Mamba-based PTRnetPure.
    Embeds the 4 sequence features + tissue exactly as before, masks,
    then runs through a Conv1d input layer + 10 ConvBlocks + 2-layer head.
    """
    def __init__(self, config: DictConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.max_seq_length = config.max_seq_length

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0

        # ---- begin conv-tower ----
        self.input_dim = config.dim_embedding_token
        self.conv_in = nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, stride=1)
        self.blocks = nn.ModuleList([ConvBlockRiboNN(self.input_dim, config.dropout) for _ in range(config.num_layers)])
        # ---- end conv-tower ----

        self.predictor: Optional[Predictor] = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs = (rna_data, tissue_id, seq_lengths, codon_freqs)
        returns: (batch, 1)
        """
        rna_data, tissue_id = inputs[0], inputs[1]

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        # Permute seq_embedding to match Conv1d input shape
        x = x.permute(0, 2, 1)  # (batch_size, dim_embedding_token, max_seq_length)
        # (B, E_s, L)
        x = self.conv_in(x)                                                # (B, E_s, L-4)
        for blk in self.blocks:
            x = blk(x)                                                     # (B, E_s, L_out)

        # 3) head
        # flatten
        x = x.flatten(start_dim=1)                                         # (B, E_s * L_out)

        # lazy-init predictor network based on flattened size
        if self.predictor is None:
            in_dim = x.size(1)
            self.predictor = Predictor(self.config, input_size=in_dim).to(self.device)

        probs = self.predictor(x)  # (B, 1)

        return probs


if __name__ == "__main__":
    # Test forward pass
    config_dev = OmegaConf.load("config/RiboNN.yml")
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

    model = RiboNN(config_dev, torch.device("cpu"))

    print(model(sample_batch))
