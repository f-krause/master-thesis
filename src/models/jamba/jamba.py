# TODO try out jamba
#  otherwise uninstall "pip uninstall jamba griffin-torch"
#  (pip install jamba installed (and uninstalled) a lot of packages)



import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from jamba.model import JambaBlock
from griffin_torch.main import GriffinResidualBlock


import gc
gc.collect()
torch.cuda.empty_cache()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class ModelJamba(nn.Module):
    """
    Jamba model implementation.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int): Number of experts.
        num_experts_per_tok (int): Number of experts per token.
        pre_emb_norm (bool): Whether to normalize the embeddings.
        return_embeddings (bool): Whether to return the embeddings.
        layers (nn.ModuleList): List of JambaBlock layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.

    """

    def __init__(self, config: DictConfig, device: torch.device):
        super(ModelJamba, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token
        self.embedding_dim = self.dim_embedding_tissue + self.dim_embedding_token

        # Embedding layers
        # TODO embedding layer norm?
        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue, max_norm=self.max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)

        # Jamba Layers
        self.model_layers = nn.ModuleList(
            [
                JambaBlock(
                    self.embedding_dim,
                    config.d_state,
                    config.d_conv,
                    config.heads,
                    config.num_experts,
                    config.num_experts_per_token,
                )
                for _ in range(config.depth)
            ]
        )

        # Griffin Layers
        # self.model_layers = nn.ModuleList().to(self.device)
        # self.model_layers.append(
        #     GriffinResidualBlock(
        #         self.embedding_dim,
        #         2,
        #         4,
        #         0.1,
        #         config.heads,
        #     ).to(self.device)
        # )

        self.predictor = Predictor(config, self.embedding_dim).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        # Expand tissue embedding to match sequence length
        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).repeat(1, seq_embedding.size(1),
                                                                         1)  # (batch_size, seq_len, dim_embedding_token)

        # Concatenate sequence embedding and tissue embedding
        combined_embedding = torch.cat((seq_embedding, tissue_embedding_expanded),
                                       dim=2)  # (batch_size, seq_len, embedding_dim)

        # Create attention mask for padding positions
        attention_mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        combined_embedding = combined_embedding * attention_mask

        x = combined_embedding.to(self.device)

        # Apply Jamba layers
        for layer in self.model_layers:
            x = layer(x)  # Jamba: FIXME cuda OOM already with first call of layer()
            # x = layer(x) + x  # Griffin: FIXME RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2)))  # (batch_size, 1, embedding_dim)
        out_last = x.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        y_pred = self.predictor(out_last)

        return y_pred
