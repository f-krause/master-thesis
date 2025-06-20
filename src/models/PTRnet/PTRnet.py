import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from utils.knowledge_db import TISSUES, CODON_MAP_DNA
from utils.utils import check_config
from models.predictor import Predictor
from data_handling.train_data_seq import TOKENS


# word to vec approach
# https://github.com/mat310/W2VC/blob/master/code/wanmei2.ipynb
class KMerEmbedding(nn.Module):
    # LEGACY
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
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

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


class ConvEncoderRiboNN(nn.Module):
    def __init__(self, nr_channels, config):
        super().__init__()
        self.conv_in = nn.Conv1d(nr_channels, nr_channels, kernel_size=5, stride=1, padding=2)
        self.blocks = nn.ModuleList(
            [ConvBlockRiboNN(nr_channels, config.predictor_dropout)
             for _ in range(config.num_layers)]
        )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return x


class DeconvBlockRiboNN(nn.Module):
    """
    One “inverse” conv‐block for decoding:
      1) Upsample by factor 2 (undoes MaxPool1d(kernel_size=2))
      2) ConvTranspose1d(kernel_size=5) (undoes Conv1d(kernel_size=5))
      3) ReLU
      4) LayerNorm over channels (via transpose)
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # ConvTranspose1d with same kernel_size=5, stride=1, no padding
        self.deconv = nn.ConvTranspose1d(channels, channels, kernel_size=5, stride=1, padding=2, output_padding=0)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L′) where L′ was pooled by 2 and reduced by conv
        x = self.upsample(x)               # → (B, C, 2·L′)
        x = self.deconv(x)                 # → (B, C, 2·L′ + 4)
        x = self.act(x)
        # apply LayerNorm on channels → need to swap (B, L, C)
        x = x.transpose(1, 2)              # → (B, L_out, C)
        x = self.norm(x)
        x = x.transpose(1, 2)              # → (B, C, L_out)
        return x


class ConvDecoderRiboNN(nn.Module):
    """
    Decoder that mirrors:
        conv_in → [ ConvBlockRiboNN ] * num_layers
    by using:
        [ DeconvBlockRiboNN ] * num_layers → conv_out
    """
    def __init__(self, nr_channels: int, num_layers: int) -> None:
        super().__init__()
        # Build one DeconvBlock per encoder block, in reverse order
        self.blocks = nn.ModuleList(
            [DeconvBlockRiboNN(nr_channels) for _ in range(num_layers)]
        )
        # Final ConvTranspose1d to “undo” the initial conv_in(kernel_size=5)
        self.conv_out = nn.ConvTranspose1d(nr_channels, nr_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x: torch.Tensor, L_target) -> torch.Tensor:
        # x: (B, C, L_latent), output of the last encoder block
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)   # → (B, C, original_length)
        if x.size(2) > L_target:
            x = x[:, :, :L_target]
        elif x.size(2) < L_target:
            pad = L_target - x.size(2)
            x = nn.functional.pad(x, (0, pad))  # pad on the right
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
                if config.dim_embedding_tissue > 32:
                    raise Exception(
                        f"WARNING: concatenating seq and tissue embedding, tissue embedding seems too large {config.dim_embedding_tissue}")
            else:
                nr_channels = config.dim_embedding_token
        elif config.seq_encoding == "word2vec":
            raise NotImplementedError("Word2Vec encoding is not implemented in this version.")
            # self.seq_encoder = KMerEmbedding().to(device)
            # nr_channels = 64
        elif config.seq_encoding == "ohe":
            # Just pass through OHE data
            self.seq_encoder = nn.Identity()
            nr_channels = 23 + config.dim_embedding_tissue
        else:
            raise ValueError(f"Unknown sequence encoding: {config.seq_encoding}")

        if self.all_tissues and not self.pretrain:
            self.tissue_encoder = nn.Embedding(
                num_embeddings=len(TISSUES),
                embedding_dim=config.dim_embedding_tissue,
                max_norm=config.embedding_max_norm,
            )

        self.encoder = ConvEncoderRiboNN(nr_channels=nr_channels, config=config).to(device)

        # ---- head (lazy init) ----
        if self.pretrain:
            self.predictors: Optional[nn.ModuleList] = None
            self.decoder = ConvDecoderRiboNN(nr_channels, config.num_layers).to(device)
        else:
            self.predictor: Optional[Predictor] = None

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
        if self.all_tissues and not self.pretrain:
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
        L_target = x.size(2)
        x = self.encoder(x)                                                # (B, E_s, L_out)
        out_dim = x.size(1)

        # if pretrain
        if self.pretrain:
            # deconv-tower
            x = self.decoder(x, L_target)                                             # (B, E_s, L_out)
            x = x.permute(0, 2, 1)                                                    # (B, L_out, E_s)

            if self.predictors is None:
                self.predictors = nn.ModuleList(
                    [
                        nn.Linear(out_dim, 4 + 1),  # sequence_ohe + padding token
                        nn.Linear(out_dim, 5 + 1),  # coding_area_ohe
                        nn.Linear(out_dim, 3 + 1),  # sec_struc_ohe
                        nn.Linear(out_dim, 7 + 1),  # loop_type_ohe
                    ]
                ).to(self.device)

            probs = []
            for predictor in self.predictors:
                logits = predictor(x)
                probs.append(F.softmax(logits, dim=-1))

            return probs
        else:
            # flatten
            x = x.flatten(start_dim=1)  # (B, E_s * L_out)
            out_dim = x.size(1)

            # optionally concat codon freqs
            if self.frequency_features:
                x = torch.cat([x, codon_freqs], dim=1)  # (B, E_s*L_out + F)
                out_dim = x.size(1)

            # lazy-init predictor network based on flattened size
            if self.predictor is None:
                self.predictor = Predictor(self.config, input_size=out_dim).to(self.device)

            return self.predictor(x)  # (B, 1)


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
        "max_seq_length": 13637,
        "binary_class": True,
        "embedding_max_norm": 0.5,
        "gpu_id": 0,
        "nr_folds": 1,
        "tissue_id": -1,

        "pretrain": True,
        "align_aug": True,
        "random_reverse": False,
        "concat_tissue_feature": False,
        "frequency_features": True,
        "seq_only": False,
        "seq_encoding": "embedding",
    })

    config_dev = check_config(config_dev)

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
            seq = torch.randint(0, 2, (length, 23)).float()
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
