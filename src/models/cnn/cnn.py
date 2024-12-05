import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from knowledge_db import TISSUES, CODON_MAP_DNA
from models.predictor import Predictor


class ModelCNN(nn.Module):
    def __init__(self, config: DictConfig, device: torch.device):
        super(ModelCNN, self).__init__()

        self.device = device
        self.max_seq_length = config.max_seq_length  # Should be set to 10500 as per the CNN architecture

        self.tissue_encoder = nn.Embedding(len(TISSUES), config.dim_embedding_tissue,
                                           max_norm=config.embedding_max_norm)  # 29 tissues
        self.seq_encoder = nn.Embedding(len(CODON_MAP_DNA) + 1, config.dim_embedding_token, padding_idx=0,
                                        max_norm=config.embedding_max_norm)  # 64 codons + padding 0

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=config.dim_embedding_token,
            out_channels=config.num_kernels_conv1,
            kernel_size=config.kernel_size_conv1,
            stride=config.stride_conv1,
            padding=config.padding
        )
        self.pool1 = nn.MaxPool1d(kernel_size=config.max_pool1)

        self.conv2 = nn.Conv1d(
            in_channels=config.num_kernels_conv1,
            out_channels=config.num_kernels_conv2,
            kernel_size=config.kernel_size_conv2,
            stride=config.stride_conv2,
            padding=config.padding
        )
        self.pool2 = nn.MaxPool1d(kernel_size=config.max_pool2)

        # Calculate the size after convolution and pooling
        length_after_conv1 = self.max_seq_length  # Since padding='same'
        length_after_pool1 = length_after_conv1 // config.max_pool1

        length_after_conv2 = length_after_pool1  # Since padding='same'
        length_after_pool2 = length_after_conv2 // config.max_pool2

        predictor = Predictor(config, length_after_pool2 * config.num_kernels_conv2)
        self.predictor = predictor.to(self.device)

    def forward(self, inputs):
        rna_data, tissue_id = inputs[0], inputs[1]

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, padded_seq_length, dim_embedding_token)

        tissue_embedding_expanded = tissue_embedding.unsqueeze(1).expand(-1, seq_embedding.size(1), -1)

        x = seq_embedding + tissue_embedding_expanded  # (batch_size, padded_seq_length, dim_embedding_token)

        mask = (rna_data_pad != 0).unsqueeze(-1).to(self.device)
        x *= mask

        # Permute seq_embedding to match Conv1d input shape
        x = x.permute(0, 2, 1)  # (batch_size, dim_embedding_token, max_seq_length)

        # Convolutional Layer 1
        x = self.conv1(x)  # (batch_size, numFiltersConv1, max_seq_length)
        x = F.relu(x)
        x = self.pool1(x)  # (batch_size, numFiltersConv1, length_after_pool1)

        # Convolutional Layer 2
        x = self.conv2(x)  # (batch_size, numFiltersConv2, length_after_pool1)
        x = F.relu(x)
        x = self.pool2(x)  # (batch_size, numFiltersConv2, length_after_pool2)

        x = x.view(x.size(0), -1)  # (batch_size, flatten_size)

        y_pred = self.predictor(x)  # (batch_size, 1)

        return y_pred


if __name__ == "__main__":
    # Test forward pass
    config_dev = OmegaConf.load("config/cnn.yml")
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

    model = ModelCNN(config_dev, torch.device("cpu"))

    print(model(sample_batch))
