import torch.nn as nn
from torch import Tensor

from omegaconf import DictConfig


class Predictor(nn.Module):
    def __init__(self, config: DictConfig, input_size: int):
        """ Final head to predict the output (PTR ratio) from the model output """
        super().__init__()

        layers = [
            nn.LayerNorm(input_size),
            nn.Linear(input_size, config.predictor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.predictor_dropout),
            nn.Linear(config.predictor_hidden_dim, 1)
        ]

        # FIXME
        # if config.binary_class:
        #     layers.append(nn.Sigmoid())

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)
