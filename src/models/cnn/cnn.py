import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

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
        # TODO try OHE instead of Embedding!

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=config.dim_embedding_token,
            out_channels=config.num_filters_conv1,
            kernel_size=config.filter_len_conv1,
            stride=config.stride_conv1,
            padding='same'
        )
        self.pool1 = nn.MaxPool1d(kernel_size=config.max_pool1)

        self.conv2 = nn.Conv1d(
            in_channels=config.numFiltersConv1,
            out_channels=config.numFiltersConv2,
            kernel_size=config.filterLenConv2,
            stride=config.stride_conv2,
            padding='same'
        )
        self.pool2 = nn.MaxPool1d(kernel_size=config.max_pool2)

        # Calculate the size after convolution and pooling
        length_after_conv1 = self.max_seq_length  # Since padding='same'
        length_after_pool1 = length_after_conv1 // config.maxPool1

        length_after_conv2 = length_after_pool1  # Since padding='same'
        length_after_pool2 = length_after_conv2 // config.maxPool2

        predictor = Predictor(config, length_after_pool2 * config.numFiltersConv2 + config.dim_embedding_tissue)
        self.predictor = predictor.to(self.device)

    def forward(self, inputs):
        rna_data, tissue_id = inputs[0], inputs[1]

        rna_data_pad = F.pad(rna_data, (0, self.max_seq_length - rna_data.size(1)), value=0)

        # Embeddings
        tissue_embedding = self.tissue_encoder(tissue_id)  # (batch_size, dim_embedding_tissue)
        seq_embedding = self.seq_encoder(rna_data_pad)  # (batch_size, max_seq_length, dim_embedding_token)

        # Permute seq_embedding to match Conv1d input shape
        x = seq_embedding.permute(0, 2, 1)  # (batch_size, dim_embedding_token, max_seq_length)

        # Convolutional Layer 1
        x = self.conv1(x)  # (batch_size, numFiltersConv1, max_seq_length)
        x = F.relu(x)
        x = self.pool1(x)  # (batch_size, numFiltersConv1, length_after_pool1)

        # Convolutional Layer 2
        x = self.conv2(x)  # (batch_size, numFiltersConv2, length_after_pool1)
        x = F.relu(x)
        x = self.pool2(x)  # (batch_size, numFiltersConv2, length_after_pool2)

        x = x.view(x.size(0), -1)  # (batch_size, flatten_size)

        x = torch.cat((tissue_embedding, x), dim=1)  # (batch_size, flatten_size + dim_embedding_tissue)

        y_pred = self.predictor(x)  # (batch_size, 1)

        return y_pred
