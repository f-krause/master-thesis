# MambaVision: https://github.com/NVlabs/MambaVision/blob/main/mambavision/assets/arch.png
# combine transformer and mamba in parallel?


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig, OmegaConf
from knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from mamba_ssm import Mamba, Mamba2
from data_handling.train_data_seq import TOKENS


class PTRnet(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(PTRnet, self).__init__()

        if config.gpu_id != 0 and config.model.lower() == "mamba2":
            raise Exception("Currently Mamba2 only supports the default GPU (cuda:0)!")

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.pretrain = config.pretrain
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token

        # Embedding layers
        nr_tokens = len(TOKENS) + 1  # +1 for padding
        if self.pretrain: nr_tokens += 1  # +1 for MASK token

        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_tissue, max_norm=config.embedding_max_norm)
        self.seq_encoder = nn.Embedding(nr_tokens, self.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)

        # Mamba model (roughly 3 * expand * d_model^2 parameters)
        self.mamba_layers = nn.ModuleList(
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

        if self.pretrain:
            self.predictors = nn.ModuleList(
                [
                    nn.Linear(self.dim_embedding_token, 4),  # sequence_ohe
                    nn.Linear(self.dim_embedding_token, 5),  # coding_area_ohe
                    nn.Linear(self.dim_embedding_token, 3),  # sec_struc_ohe
                    nn.Linear(self.dim_embedding_token, 7),  # loop_type_ohe
                ]
            )

        else:
            self.predictor = Predictor(config, self.dim_embedding_token).to(self.device)

    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        # rna_data.append(torch.tensor([sequence_ohe, coding_area_ohe, sec_struc_ohe, loop_type_ohe]))  # 4 x n
        rna_data_pad, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, seq_len, dim_embedding_token)

        # sum up the embeddings of the 4 features (nucleotide, coding_area, sec_struc, loop_type)
        seq_embedding = seq_embedding.sum(dim=2)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        padding_mask = (seq_embedding != 0).to(self.device)
        x *= padding_mask

        # Apply Mamba model
        for layer in self.mamba_layers:
            x = layer(x)

        if self.pretrain:
            y_pred = []
            for predictor in self.predictors:
                logits = predictor(x)
                y_pred.append(F.softmax(logits, dim=-1))
        else:
            # Extract outputs corresponding to the last valid time step
            idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2)))  # (batch_size, 1, embedding_dim)
            x_last = x.gather(1, idx).squeeze(1)  # (batch_size, embedding_dim)

            y_pred = self.predictor(x_last)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    device = torch.device("cuda:0")
    config_dev = OmegaConf.load("config/PTRnet.yml")
    config_dev = OmegaConf.merge(config_dev, {"binary_class": True, "max_seq_length": 1000,
                                              "embedding_max_norm": 2, "gpu_id": 0, "pretrain": True})

    sample_batch = [
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, len(TOKENS) + 1, (config_dev.batch_size, config_dev.max_seq_length, 4)),
                                        batch_first=True),  # rna_data_padded (batch_size x max_seq_length)
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size,
                     dtype=torch.int64)  # seq_lengths (batch_size x 1)
    ]

    sample_batch = [tensor.to(torch.device(device)) for tensor in sample_batch]

    model = PTRnet(config_dev, device).to(device)

    print(model(sample_batch))
