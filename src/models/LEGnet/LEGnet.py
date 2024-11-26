# Based on https://github.com/autosome-ru/LegNet/blob/main/legnet/model.py (MIT license)
import torch
from torch import nn
import torch.nn.functional as F
from tltorch import TRL
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf

from knowledge_db import TISSUES, CODON_MAP_DNA
from models.predictor import Predictor


class Bilinear(nn.Module):
    """
    Bilinear layer introduces pairwise product to a NN to model possible combinatorial effects.
    This particular implementation attempts to leverage the number of parameters via low-rank tensor decompositions.

    Parameters
    ----------
    n : int
        Number of input features.
    out : int, optional
        Number of output features. If None, assumed to be equal to the number of input features. The default is None.
    rank : float, optional
        Fraction of maximal to rank to be used in tensor decomposition. The default is 0.05.
    bias : bool, optional
        If True, bias is used. The default is False.

    """

    def __init__(self, n: int, out=None, rank=0.05, bias=False):
        super().__init__()
        if out is None:
            out = (n,)
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))


class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension.

    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.

    """

    def __init__(self, module: nn.Module, dim=-1):
        super().__init__()
        self.mod = module
        self.dim = dim

    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)


class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.

    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.

    """

    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), int(inp // reduction)),
            Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction) + int(inp // reduction // 2), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LEGnet(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')

    def __init__(self, config: DictConfig, device: torch.device):
        super().__init__()

        self.device = device
        self.binary_class = config.binary_class
        self.block_sizes = config.block_sizes
        self.kernel_size = config.kernel_size
        self.resize_factor = config.resize_factor
        self.se_reduction = config.se_reduction
        self.final_ch = config.final_ch
        self.bn_momentum = config.bn_momentum
        self.max_seq_length = config.max_seq_length  # Should be set to 10500 as per the CNN architecture
        if config.activation.lower() == "silu":
            activation = nn.SiLU
        elif config.activation.lower() == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("Specified activation function currently not supported (only SiLU and GeLU)")

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0

        predictor = Predictor(config, config.final_ch)
        self.predictor = predictor.to(self.device)

        seqextblocks = OrderedDict()

        block = nn.Sequential(
            nn.Conv1d(
                in_channels=config.dim_embedding_token,
                out_channels=self.block_sizes[0],
                kernel_size=self.kernel_size,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(self.block_sizes[0],
                           momentum=self.bn_momentum),
            activation()  # Exponential(block_sizes[0]) #activation()
        )
        seqextblocks[f'blc0'] = block

        for ind, (prev_sz, sz) in enumerate(zip(self.block_sizes[:-1], self.block_sizes[1:])):
            block = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv1d(
                    in_channels=prev_sz,
                    out_channels=sz * self.resize_factor,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz * self.resize_factor), #activation(),

                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=sz * self.resize_factor,
                    kernel_size=self.kernel_size,
                    groups=sz * self.resize_factor // config.filter_per_group,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz * self.resize_factor), #activation(),
                SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                # nn.Dropout(0.1),
                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=prev_sz,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(prev_sz,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz), #activation(),

            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * prev_sz,
                    out_channels=sz,
                    kernel_size=self.kernel_size,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz,
                               momentum=self.bn_momentum),
                activation(),  # Exponential(sz), #activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=self.block_sizes[-1],
                out_channels=self.final_ch,
                kernel_size=1,
                padding='same',
            ),
            activation()
        )

        self.register_buffer('bins', torch.arange(start=0, end=18, step=1, requires_grad=False))

    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)

        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x

    def forward(self, inputs):
        rna_data, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        x = x.permute(0, 2, 1)  # (batch_size, dim_embedding_token, max_seq_length)  = (4, 16, 2700)

        f = self.feature_extractor(x)
        x = self.mapper(f)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)

        y_pred = self.predictor(x)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    config_dev = OmegaConf.load("config/LEGnet.yml")
    config_dev = OmegaConf.merge(config_dev,
                                 {"binary_class": True, "max_seq_length": 2700, "embedding_max_norm": 2})

    sample_batch = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (config_dev.batch_size, config_dev.max_seq_length)),
                                        batch_first=True),
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size, dtype=torch.int64)
        # seq_lengths (batch_size x 1)
    ]

    model = LEGnet(config_dev, torch.device("cpu"))

    print(model(sample_batch))
