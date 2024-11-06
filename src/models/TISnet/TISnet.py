# Based on https://github.com/huangwenze/TISnet/tree/main/tisnet
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from knowledge_db import TISSUES, CODON_MAP_DNA
from models.TISnet.resnet import ResidualBlock1D, ResidualBlock2D
from models.TISnet.se import SEBlock


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class TISnet(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device, mode="pu"):
        super(TISnet, self).__init__()

        # FIXME vvv
        self.mode = mode
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"
        # FIXME ^^^

        self.device = device
        self.binary_class = config.binary_class
        self.max_seq_length = config.max_seq_length  # Should be set to 10500 as per the CNN architecture

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0
        # TODO try OHE instead of Embedding!

        base_channel = 8
        self.conv = Conv2d(1, base_channel, kernel_size=(11, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p))
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8 + config.dim_embedding_tissue, 1)
        if self.binary_class:
            self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Tensor) -> Tensor:
        """[forward]

        Args:
            input ([tensor],N,C,W,H): input features
            Batch x Channel x Width x Height ???
        """

        rna_data, tissue_id, seq_lengths = inputs[0], inputs[1], inputs[2]

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        # Permute seq_embedding to match Conv1d input shape
        x = seq_embedding.permute(0, 2, 1)  # (batch_size, dim_embedding_token, max_seq_length)
        x = x.unsqueeze(1)
        # if self.mode == "seq":
        #     input = input[:, :, :, :4]
        # elif self.mode == "str":
        #     input = input[:, :, :, 4:]

        x = self.conv(x)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], -1)  # doc: x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])

        x = torch.cat((tissue_embedding, x), dim=1)  # (batch_size, flatten_size + dim_embedding_tissue)

        x = self.fc(x)

        if self.binary_class:
            x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    # Test model
    config_dev = OmegaConf.load("config/TISnet.yml")
    config_dev = OmegaConf.merge(config_dev,
                                 {"binary_class": True, "max_seq_length": 2700, "embedding_max_norm": 2})
    device_dev = torch.device("cpu")

    sample_batch = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (config_dev.batch_size, config_dev.max_seq_length)), batch_first=True),
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config_dev.max_seq_length] * config_dev.batch_size, dtype=torch.int64)
        # seq_lengths (batch_size x 1)
    ]

    model = TISnet(config_dev, device_dev)

    print(model(sample_batch))
