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
        self.frequency_features = config.frequency_features
        self.pretrain = config.pretrain
        self.dim_embedding_tissue = config.dim_embedding_tissue
        self.dim_embedding_token = config.dim_embedding_token

        if self.frequency_features:
            self.output_dim = self.dim_embedding_token + len(CODON_MAP_DNA)
        else:
            self.output_dim = self.dim_embedding_token

        # Embedding layers of input features
        nr_tokens = len(TOKENS) + 2  # + 1 for padding, one for pretraining MASK token
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
                    nn.Linear(self.output_dim, 4),  # sequence_ohe
                    nn.Linear(self.output_dim, 5),  # coding_area_ohe
                    nn.Linear(self.output_dim, 3),  # sec_struc_ohe
                    nn.Linear(self.output_dim, 7),  # loop_type_ohe
                ]
            )
        else:
            self.predictor = Predictor(config, self.output_dim).to(self.device)

    def forward(self, inputs: Tensor, mask: Tensor = None) -> Tensor:
        rna_data, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_token)
        seq_embedding = self.seq_encoder(rna_data)  # (batch_size, seq_len, dim_embedding_token)

        # sum up the embeddings of the 4 features (nucleotide, coding_area, sec_struc, loop_type)
        seq_embedding = seq_embedding.sum(dim=2)

        # contextualize with tissue embedding
        x = seq_embedding + tissue_embedding.unsqueeze(1)  # (batch_size, padded_seq_length, dim_embedding_token)

        # Apply padding mask to zero out tissue embedding for padded tokens
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
            if self.frequency_features:
                x_last = torch.cat([x_last, inputs[3]], dim=-1)
            y_pred = self.predictor(x_last)

        return y_pred


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
