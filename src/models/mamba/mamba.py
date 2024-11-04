import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from mamba_ssm import Mamba, Mamba2


class ModelMamba(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(ModelMamba, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token
        self.embedding_dim = self.dim_embedding_tissue + self.dim_embedding_token

        # Embedding layers
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_tissue, max_norm=config.embedding_max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, self.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)

        # Mamba model
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.embedding_dim,  # Model dimension d_model
            d_state=config.d_state,  # SSM state expansion factor
            d_conv=config.d_conv,  # Local convolution width
            expand=config.expand,  # Block expansion factor
            # headdim=config.headdim  # TODO only for mamba-2
        ).to(self.device)

        self.predictor = Predictor(config, self.embedding_dim).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        # Expand tissue embedding to match sequence length (batch_size, seq_len, dim_embedding_token)
        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).repeat(1, seq_embedding.size(1), 1)

        # Concat embeddings (batch_size, seq_len, embedding_dim)
        combined_embedding = torch.cat((seq_embedding, tissue_embedding_expanded), dim=2)

        # Apply Mamba model
        out = self.mamba(combined_embedding)  # (batch_size, seq_len, embedding_dim)
        # out = self.mamba(combined_embedding, cu_seqlens=seq_lengths) # TODO for mamba2

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)))  # (batch_size, 1, embedding_dim)
        out_last = out.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        y_pred = self.predictor(out_last)

        return y_pred
