import torch
import torch.nn as nn
from torch import Tensor
from box import Box
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from mamba_ssm import Mamba2, Mamba


class ModelMamba(nn.Module):
    def __init__(self, config: Box, device: torch.device):
        super(ModelMamba, self).__init__()

        self.device = device
        self.max_norm = 2
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_token = config.dim_embedding_token
        self.tissue_embedding_dim = config.tissue_embedding_dim
        self.embedding_dim = self.dim_embedding_token + self.tissue_embedding_dim

        # Embedding layers
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.tissue_embedding_dim, max_norm=self.max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, self.dim_embedding_token, padding_idx=0,
                                        max_norm=self.max_norm)

        # Mamba model
        # TODO Mamba2 throws: "Triton Error [CUDA]: device kernel image is invalid"
        self.mamba = Mamba(
            d_model=self.embedding_dim,  # Model dimension d_model
            d_state=config.d_state,  # SSM state expansion factor
            d_conv=config.d_conv,  # Local convolution width
            expand=config.expand,  # Block expansion factor
            # headdim=config.headdim  #  only for mamba-2
        ).to(self.device)

        # Predictor network
        self.predictor = Predictor(self.embedding_dim, config.out_hidden_size).to(self.device)
        # self.predictor = Predictor(self.embedding_dim + self.tissue_embedding_dim, config.out_hidden_size).to(
        #     self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]
        tissue_id = torch.tensor(tissue_id).to(self.device)
        rna_data_pad = rna_data_pad.to(self.device)
        seq_lengths = torch.tensor(seq_lengths).to(self.device)

        # Embedding layers
        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, tissue_embedding_dim)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        # Expand tissue embedding to match sequence length
        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).repeat(1, seq_embedding.size(1),
                                                                         1)  # (batch_size, seq_len, tissue_embedding_dim)

        # Concatenate sequence embedding and tissue embedding
        combined_embedding = torch.cat((seq_embedding, tissue_embedding_expanded),
                                       dim=2)  # (batch_size, seq_len, embedding_dim)

        # Create attention mask for padding positions
        attention_mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        combined_embedding = combined_embedding * attention_mask

        # Apply Mamba2 model
        x = combined_embedding.to(self.device)
        out = self.mamba(x)  # (batch_size, seq_len, embedding_dim)

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)))  # (batch_size, 1, embedding_dim)
        out_last = out.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        # combined_features = torch.cat((out_last, tissue_embedding), dim=1)  # add tissue embedding to output
        # seemed to worsen results

        y_pred = self.predictor(out_last)

        return y_pred
