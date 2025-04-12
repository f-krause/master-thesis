import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from mamba_ssm import Mamba, Mamba2


class ModelMamba(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device, model: str = "mamba"):
        super(ModelMamba, self).__init__()

        if config.gpu_id != 0 and config.model.lower() == "mamba2":
            raise Exception("Currently Mamba2 only supports the default GPU (cuda:0)!")

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token
        self.model = model

        # Embedding layers
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_tissue, max_norm=config.embedding_max_norm)
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, self.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)

        # Mamba model (roughly 3 * expand * d_model^2 parameters)
        if self.model.lower() == 'mamba':
            self.model_layers = nn.ModuleList(
                [
                    Mamba(
                        d_model=self.dim_embedding_token,  # Model dimension d_model
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=config.d_conv,  # Local convolution width
                        expand=config.expand,  # Block expansion factor
                    ).to(self.device)
                    for _ in range(config.num_layers)
                ]
            )
        elif self.model.lower() == 'mamba2':
            self.model_layers = nn.ModuleList(
                [
                    Mamba2(
                        d_model=self.dim_embedding_token,  # Model dimension d_model
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=config.d_conv,  # Local convolution width
                        expand=config.expand,  # Block expansion factor
                        headdim=config.head_dim
                    ).to(self.device)
                    for _ in range(config.num_layers)
                ]
            )
        else:
            raise ValueError(f"Mamba model {model} not supported")

        self.predictor = Predictor(config, self.dim_embedding_token).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        # Apply Mamba model
        if self.model.lower() == 'mamba':
            for layer in self.model_layers:
                x = layer(x)
        elif self.model.lower() == 'mamba2':
            for layer in self.model_layers:
                x = layer(x, cu_seqlens=seq_lengths)
        else:
            raise ValueError(f"Model type {type(self.mamba)} not supported")

        # Extract outputs corresponding to the last valid time step
        idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2)))  # (batch_size, 1, embedding_dim)
        x_last = x.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

        y_pred = self.predictor(x_last)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    device = torch.device("cuda:0")
    config_dev = OmegaConf.load("config/mamba.yml")
    config_dev = OmegaConf.merge(config_dev,
                                 {"binary_class": True, "max_seq_length": 2700, "embedding_max_norm": 2, "gpu_id": 0})

    sample_batch = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (config_dev.batch_size, config_dev.max_seq_length)),
                                        batch_first=True),
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size, dtype=torch.int64)
        # seq_lengths (batch_size x 1)
    ]

    sample_batch = [tensor.to(torch.device(device)) for tensor in sample_batch]

    model = ModelMamba(config_dev, torch.device("cuda:0")).to(device)

    print(model(sample_batch))
