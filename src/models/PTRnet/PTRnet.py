import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from data_handling.train_data_seq import TOKENS


# TODO
# Idea: put together what you think works, then tune it fully (also predictor layer and much more stuff!
# Then do proper ablation study
# also tune freq model for fairness
# hopefully it will be beaten!


# word to vec approach
# https://github.com/mat310/W2VC/blob/master/code/wanmei2.ipynb
class KMerEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        with open("/export/share/krausef99dm/data/w2v_model_data.pkl", 'rb') as f:
            model_data = pickle.load(f)
        num_kmers, embedding_dim = model_data["embedding_matrix"].shape
        self.embedding = nn.Embedding(num_kmers, embedding_dim)
        self.embedding.weight.data.copy_(model_data["embedding_matrix"])
        self.embedding.weight.requires_grad = False  # Freeze embeddings with False

    def forward(self, kmer_indices):
        return self.embedding(kmer_indices)


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


class PTRnetRiboNN(nn.Module):
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
        self.frequency_features = config.frequency_features
        self.seq_only = config.seq_only
        self.pretrain = config.pretrain
        self.all_tissues = config.tissue_id < 0

        if config.seq_encoding == "embedding":
            nr_tokens = len(TOKENS) + 2
            self.seq_encoder = nn.Embedding(nr_tokens, config.dim_embedding_token, padding_idx=0,
                                            max_norm=config.embedding_max_norm)
            if self.config.concat_tissue_feature:
                nr_channels = config.dim_embedding_token + config.dim_embedding_tissue
                if config.dim_embedding_tissue > 16:
                    raise Exception(
                        f"WARNING: concatenating seq and tissue embedding, tissue embedding seems too large {config.dim_embedding_tissue}")
            else:
                nr_channels = config.dim_embedding_token
        elif config.seq_encoding == "word2vec":
            self.seq_encoder = KMerEmbedding().to(device)
            nr_channels = 64  # FIXME verify
        elif config.seq_encoding == "ohe":
            # Just pass through OHE data
            self.seq_encoder = nn.Identity()
            nr_channels = 23 + config.dim_embedding_tissue
        else:
            raise ValueError(f"Unknown sequence encoding: {config.seq_encoding}")

        if self.all_tissues:
            self.tissue_encoder = nn.Embedding(
                num_embeddings=len(TISSUES),
                embedding_dim=config.dim_embedding_tissue,
                max_norm=config.embedding_max_norm,
            )

        # ---- begin conv-tower ----
        self.conv_in = nn.Conv1d(nr_channels, nr_channels, kernel_size=5, stride=1)
        self.blocks = nn.ModuleList(
            [ConvBlockRiboNN(nr_channels, config.predictor_dropout) for _ in range(config.num_layers)]
        )
        # ---- end conv-tower ----

        self.predictor: Optional[Predictor] = None  # lazy-init predictor network

        # if someone still flips on pretrain, we haven’t implemented that here
        if self.pretrain:
            raise NotImplementedError("Conv-based PTRnetRiboNN does not support pretrain mode.")

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs = (rna_data, tissue_id, seq_lengths, codon_freqs)
        returns: (batch, 1)
        """
        rna_data, tissue_id, seq_lengths, codon_freqs = inputs[0], inputs[1], inputs[2], inputs[3]

        rna_data_pad = F.pad(rna_data, (0, 0, 0, self.max_seq_length - rna_data.size(1)), value=0)  # (B, L, 4)

        # 1) embed sequence + tissue exactly as before
        x = self.seq_encoder(rna_data_pad)                         # (B, L, 4, E_s)

        if self.config.seq_encoding == "embedding":
            if self.seq_only:
                # pick only nucleotide channel
                x = x[:, :, 0, :]                                  # (B, L, E_s)
            else:
                # sum the four feature‐channels
                x = x.sum(dim=2)                                   # (B, L, E_s)

        # add tissue
        if self.all_tissues:
            tissue_embedding = self.tissue_encoder(tissue_id)  # (B, E_t)
            if self.config.concat_tissue_feature or self.config.seq_encoding == "ohe":
                tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
                x = torch.cat([x, tissue_embedding_expanded], dim=-1)  # (B, L, E_s + E_t)
            else:
                tissue_embedding = self.tissue_encoder(tissue_id)  # (B, E_t)
                x = x + tissue_embedding.unsqueeze(1)  # (B, L, E_s)

        # mask padding
        if not self.config.align_aug:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < seq_lengths[:, None]
            x = x * mask.unsqueeze(-1).to(x.dtype)

        # 2) conv-tower
        # move to (B, C, L)
        x = x.permute(0, 2, 1)                                             # (B, E_s, L)
        x = self.conv_in(x)                                                # (B, E_s, L-4)
        for blk in self.blocks:
            x = blk(x)                                                     # (B, E_s, L_out)

        # 3) head
        # flatten
        x = x.flatten(start_dim=1)                                         # (B, E_s * L_out)

        # optionally concat codon freqs
        if self.frequency_features:
            x = torch.cat([x, codon_freqs], dim=1)                         # (B, E_s*L_out + F)

        # lazy-init predictor network based on flattened size
        if self.predictor is None:
            in_dim = x.size(1)
            self.predictor = Predictor(self.config, input_size=in_dim).to(self.device)

        probs = self.predictor(x)  # (B, 1)

        return probs


class PTRnet(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(PTRnet, self).__init__()
        self.ptrnet = PTRnetRiboNN(config, device)
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
        "embedding_max_norm": 0.5,
        "gpu_id": 0,
        "tissue_id": -1,
        "seq_encoding": "embedding",
        "pretrain": False,
        "frequency_features": True,
        "concat_tissue_feature": False,
        "seq_only": True,
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

        if config_dev.seq_encoding == "ohe":
            seq = torch.randint(0, 2, (length, 19))
        elif config_dev.seq_encoding == "word2vec":
            seq = torch.randint(0, 64, (length,))

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

    print("Parameters:", sum(p.numel() for p in model.parameters()))
    y_pred_dev = model(sample_batch)
    print(y_pred_dev)
