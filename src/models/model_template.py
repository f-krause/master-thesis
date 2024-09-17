import torch.nn as nn


class ModelTemplate(nn.Module):
    def __init__(self):
        super(ModelTemplate, self).__init__()
        self.layer = nn.Linear(10, 2)  # Example layer

        raise NotImplementedError("Please implement the model")

    def forward(self, x):
        return self.layer(x)
