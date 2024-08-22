import torch.nn as nn


class ModelDummy(nn.Module):
    def __init__(self):
        super(ModelDummy, self).__init__()
        self.layer = nn.Linear(10, 2)  # Example layer

    def forward(self, x):
        return self.layer(x)
