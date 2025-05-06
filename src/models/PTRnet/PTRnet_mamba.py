import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from utils.knowledge_db import TISSUES, CODON_MAP_DNA

from models.predictor import Predictor
from mamba_ssm import Mamba
from data_handling.train_data_seq import TOKENS

# Idea: put together what you think works, then tune it fully (also predictor layer and much more stuff!
# Then do proper ablation study
# also tune freq model for fairness
# hopefully it will be beaten!


class CNNFeatureExtractor(nn.Module):
    # TODO
    # deep with small kernel size better than shallow with large kernels
    # maybe treat each feature of the model as one channel?
    def __init__(self, in_channels=19, conv_filters=64, kernel_size=5, stride=1, pool_size=3, pool_stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, conv_filters, kernel_size, stride, padding="same")
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(pool_size, stride=pool_stride, padding=1)
        # self.conv2 = nn.Conv1d(conv_filters, int(conv_filters * 2), kernel_size * 2, stride, padding="same")

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert (batch, seq_len, channels) → (batch, channels, seq_len)
        x = self.relu(self.conv1(x))
        # x = self.pool(x)
        # x = self.relu(self.conv2(x))  # second relu necessary?
        # x = self.conv2(x)
        return x  # (batch, conv_filters, reduced_seq_len)


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
        self.embedding.weight.requires_grad = True  # Freeze embeddings with False

    def forward(self, kmer_indices):
        return self.embedding(kmer_indices)


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
        self.input_size = self.dim_embedding_token  # + len(CODON_MAP_DNA)

        if self.frequency_features:
            self.output_dim = self.dim_embedding_token + len(CODON_MAP_DNA)
        else:
            self.output_dim = self.dim_embedding_token * config.max_seq_length  # FIXME

        self.tissue_encoder = nn.Embedding(len(TISSUES), self.dim_embedding_tissue, max_norm=config.embedding_max_norm)

        nr_tokens = len(TOKENS) + 2
        if config.seq_encoding == "embedding":
            self.seq_encoder = nn.Embedding(nr_tokens, self.dim_embedding_token, padding_idx=0,
                                            max_norm=config.embedding_max_norm)
        elif config.seq_encoding == "ohe":
            self.seq_encoder = CNNFeatureExtractor().to(device)  # FIXME CNN embedding -> even necessary? already included in
            raise NotImplementedError("implement ohe")
        elif config.seq_encoding == "word2vec":
            self.seq_encoder = KMerEmbedding().to(device)  # FIXME test word2vec embedding
        else:
            raise ValueError(f"Unknown sequence encoding: {config.seq_encoding}. Choose from embedding, ohe, word2vec.")

        # Layer norms
        if self.nr_layers > 1:
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(self.input_size) for _ in range(self.nr_layers)])
            self.layer_norm_final = nn.LayerNorm(self.input_size)

        # Mamba model
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.input_size,  # Model dimension d_model (in-/output size)
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
            self.predictor = Predictor(config, input_size=self.output_dim).to(self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        rna_data, tissue_id, seq_lengths, codon_freqs = inputs[0], inputs[1], inputs[2], inputs[3]

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)

        seq_embedding = self.seq_encoder(rna_data)  # FIXME default & word2vec embedding
        # seq_embedding = self.seq_encoder(rna_data.float()).permute(0, 2, 1)  # FIXME CNN embedding

        if self.seq_only:
            seq_embedding = seq_embedding[:, :, 0, :]
        else:
            # sum up the embeddings of the 4 features (nucleotide, coding_area, sec_struc, loop_type)
            seq_embedding = seq_embedding.sum(dim=2)  # (batch_size, seq_len, dim_embedding_token)  # FIXME default setting
            # pass

        # contextualize with tissue embedding
        x = seq_embedding + tissue_embedding.unsqueeze(1)  # (batch_size, seq_len, dim_embedding_token)

        # Apply padding mask
        # padding_mask = (seq_embedding != 0).to(self.device)  # only works for embeddings!
        padding_mask = torch.arange(x.shape[1], device=x.device)[None, :] < seq_lengths[:, None]
        padding_mask = padding_mask.unsqueeze(-1)

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
            # idx = ((seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.size(2)))
            # x_last = x.gather(1, idx).squeeze(1)
            # x_last = x.mean(dim=1)  # FIXME
            x_last = x.flatten(start_dim=1)  # FIXME
            x_last = torch.nn.functional.pad(x_last, (0, self.output_dim - x_last.shape[1]))  # FIXME

            if self.frequency_features:
                x_last = torch.cat([x_last, codon_freqs], dim=-1)
            y_pred = self.predictor(x_last)

        return y_pred


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

    def forward(self, x: Tensor) -> Tensor:
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
        self.single_tissue = config.tissue_id < 0

        # embeddings (exactly as before)
        if self.single_tissue:
            self.tissue_encoder = nn.Embedding(
                num_embeddings=len(TISSUES),
                embedding_dim=config.dim_embedding_tissue,
                max_norm=config.embedding_max_norm,
            )
        nr_tokens = len(TOKENS) + 2
        self.seq_encoder = nn.Embedding(
            num_embeddings=nr_tokens,
            embedding_dim=config.dim_embedding_token,
            padding_idx=0,
            max_norm=config.embedding_max_norm,
        )

        # ---- begin conv-tower ----
        C = config.dim_embedding_token
        D = config.predictor_dropout
        N = config.num_layers  # we will interpret num_layers=10 → 10 conv-blocks
        self.conv_in = nn.Conv1d(C, C, kernel_size=5, stride=1)
        self.blocks = nn.ModuleList([ConvBlockRiboNN(C, D) for _ in range(N)])
        # ---- end conv-tower ----

        self.predictor: Optional[Predictor] = None

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
        seq_emb = self.seq_encoder(rna_data_pad)                           # (B, L, 4, E_s)

        if self.seq_only:
            # pick only nucleotide channel
            x = seq_emb[:, :, 0, :]                                  # (B, L, E_s)
        else:
            # sum the four feature‐channels
            x = seq_emb.sum(dim=2)                                   # (B, L, E_s)

        # add tissue
        if self.single_tissue:
            tissue_emb = self.tissue_encoder(tissue_id)  # (B, E_t)
            x = x + tissue_emb.unsqueeze(1)                              # (B, L, E_s)

        # mask padding
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


class PTRnetMLP(nn.Module):
    # TODO
    pass


class PTRnet(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(PTRnet, self).__init__()
        # self.ptrnet = PTRnetPure(config, device)
        self.ptrnet = PTRnetRiboNN(config, device)  # FIXME
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
        "pretrain": False,
        "frequency_features": False,
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

        # vvv FIXME OHE setting
        # seq = torch.randint(0, 2, (length, 19))

        # vvv FIXME word2vec sequence only setting
        # seq = torch.randint(0, 64, (length,))

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
