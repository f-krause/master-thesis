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


class PTRnetPure(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(PTRnetPure, self).__init__()

        if config.gpu_id != 0 and config.model.lower() == "mamba2":
            raise Exception("Currently Mamba2 only supports the default GPU (cuda:0)!")

        self.device = device
        self.max_seq_length = config.max_seq_length
        self.frequency_features = config.frequency_features
        self.seq_only = config.seq_only
        self.pretrain = config.pretrain
        self.nr_layers = config.num_layers
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token

        if self.frequency_features:
            self.output_dim = self.dim_embedding_token + len(CODON_MAP_DNA)
        else:
            self.output_dim = self.dim_embedding_token

        nr_tokens = len(TOKENS) + 2
        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_tissue, max_norm=config.embedding_max_norm)
        self.seq_encoder = nn.Embedding(nr_tokens, self.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)

        # Layer norms
        if self.nr_layers > 1:
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(self.dim_embedding_token) for _ in range(self.nr_layers)])
            self.layer_norm_final = nn.LayerNorm(self.dim_embedding_token)

        # Mamba model
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.dim_embedding_token,  # Model dimension d_model
                    d_state=config.d_state,  # SSM state expansion factor
                    d_conv=config.d_conv,  # Local convolution width
                    expand=config.expand,  # Block expansion factor
                ).to(self.device)
                for _ in range(self.nr_layers)
            ]
        )

        if self.pretrain:
            self.predictors = nn.ModuleList(
                [
                    nn.Linear(self.output_dim, 4),  # sequence_ohe
                    nn.Linear(self.output_dim, 5),  # coding_area_ohe
                    nn.Linear(self.output_dim, 3),  # sec_struc_ohe
                    nn.Linear(self.output_dim, 7),  # loop_type_ohe
                ]
            )
        else:
            self.predictor = Predictor(config, self.output_dim).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data, tissue_id, seq_lengths, codon_freqs = inputs[0], inputs[1], inputs[2], inputs[3]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data)  # (batch_size, seq_len, 4, dim_embedding_token)

        # sum up the embeddings of the 4 features (nucleotide, coding_area, sec_struc, loop_type)
        if self.seq_only:
            seq_embedding = seq_embedding[:, :, 0, :]
        else:
            seq_embedding = seq_embedding.sum(dim=2)  # (batch_size, seq_len, dim_embedding_token)

        # contextualize with tissue embedding
        x = seq_embedding + tissue_embedding.unsqueeze(1)  # (batch_size, seq_len, dim_embedding_token)

        # Apply padding mask
        padding_mask = (seq_embedding != 0).to(self.device)
        x *= padding_mask

        # Pass through Mamba layers with residual connections and layer norm
        if self.nr_layers > 1:
            for layer, lnorm in zip(self.mamba_layers, self.mamba_norms):
                x = lnorm(x)
                x = layer(x)
            x = self.layer_norm_final(x)
        else:
            x = self.mamba_layers[0](x)

        if self.pretrain:
            y_pred = []
            for predictor in self.predictors:
                logits = predictor(x)
                y_pred.append(F.softmax(logits, dim=-1))
        else:
            # Extract last valid time step
            idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2)))
            x_last = x.gather(1, idx).squeeze(1)
            if self.frequency_features:
                x_last = torch.cat([x_last, codon_freqs], dim=-1)
            y_pred = self.predictor(x_last)

        return y_pred


class PTRnetMLP(nn.Module):
    # TODO
    pass


class PTRnet(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(PTRnet, self).__init__()
        self.ptrnet = PTRnetPure(config, device)
        # self.ptrnet = None

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ptrnet(inputs)


if __name__ == "__main__":
    # Test forward pass
    device = torch.device("cuda:0")
    config_dev = OmegaConf.load("config/PTRnet.yml")
    config_dev = OmegaConf.merge(config_dev, {
        "batch_size": 8,
        "max_seq_length": 9000,
        "binary_class": True,
        "embedding_max_norm": 2,
        "gpu_id": 0,
        "pretrain": False,
        "frequency_features": True,
        "seq_only": False,
    })

    sequences, seq_lengths = [], []
    for i in range(config_dev.batch_size):
        length = torch.randint(100, config_dev.max_seq_length + 1, (1,)).item()
        # Generate each column with its own integer range
        col0 = torch.randint(low=6, high=10, size=(length,))  # values in [6,9] - seq ohe
        col1 = torch.randint(low=1, high=6, size=(length,))  # values in [1,5] - coding area
        col2 = torch.randint(low=10, high=13, size=(length,))  # values in [10,12] - loop type pred
        col3 = torch.randint(low=13, high=20, size=(length,))  # values in [13,19] - sec structure pred

        seq = torch.stack([col0, col1, col2, col3], dim=-1)
        sequences.append(seq)
        seq_lengths.append(length)

    sample_batch = [
        torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),  # rna_data_padded (B x N x D)
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (B)
        torch.tensor(seq_lengths, dtype=torch.int64),  # seq_lengths (B)
        torch.randn(config_dev.batch_size, len(CODON_MAP_DNA)),  # frequency_features (B x 64)
    ]

    sample_batch = [tensor.to(torch.device(device)) for tensor in sample_batch]

    model = PTRnet(config_dev, device).to(device)

    print(model(sample_batch))
