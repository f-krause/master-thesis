import torch.nn as nn
from torch import Tensor


class Predictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Final head to predict the output (PTR ratio) from the model output

        Args:
            input_size: int, size of the input tensor
            hidden_size: int, size of the hidden layer
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)
