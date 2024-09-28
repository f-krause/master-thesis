# FIXME needs cuda 12.1 to run

import torch
import torch.nn as nn
from torch import Tensor
from box import Box
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class ModelXLSTM(nn.Module):
    def __init__(self, config: Box, device: torch.device):
        super(ModelXLSTM, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length

        # Embedding layers
        # TODO embedding layer norm?
        self.tissue_encoder = nn.Embedding(len(TISSUES), config.tissue_embedding_dim, max_norm=self.max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)

        # Total embedding dimension after concatenation
        self.embedding_dim = config.dim_embedding_token + config.tissue_embedding_dim

        # xLSTMBlockStack configuration
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=config.conv1d_kernel_size,
                    qkv_proj_blocksize=config.qkv_proj_blocksize,
                    num_heads=config.num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=device.type,
                    num_heads=config.num_heads,
                    conv1d_kernel_size=config.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=config.proj_factor, act_fn=config.act_fn
                ),
            ),
            context_length=config.max_seq_length,
            num_blocks=config.num_blocks,
            embedding_dim=self.embedding_dim,
            slstm_at=config.slstm_at,
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)

        self.predictor = Predictor(self.embedding_dim, config.out_hidden_size).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data_pad = rna_data_pad.to(self.device)

        # Embedding layers
        tissue_embedding = self.tissue_encoder(tissue_id)
        seq_embedding = self.seq_encoder(rna_data_pad)

        # Expand tissue embedding to match sequence length
        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).repeat(1, seq_embedding.size(1), 1)

        # Concatenate sequence embedding and tissue embedding
        combined_embedding = torch.cat((seq_embedding, tissue_embedding_expanded), dim=2)

        # Create attention mask for padding positions
        attention_mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        combined_embedding = combined_embedding * attention_mask

        # Forward pass through xLSTMBlockStack
        x = combined_embedding.to(self.device)
        out = self.xlstm_stack(x)

        # Extract outputs corresponding to the last valid time step
        seq_lengths = torch.tensor(seq_lengths).to(self.device)
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)))  # wtf is going on here
        out_last = out.gather(1, idx).squeeze(1)

        y_pred = self.predictor(out_last)

        return y_pred
