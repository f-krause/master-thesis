import torch.nn as nn


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.layer = nn.Linear(10, 2)  # Example layer

    def forward(self, x):
        return self.layer(x)
