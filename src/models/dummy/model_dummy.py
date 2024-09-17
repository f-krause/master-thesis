import torch
import torch.nn as nn


class ModelDummy(nn.Module):
    def __init__(self, device: torch.device):
        super(ModelDummy, self).__init__()
        self.device = device
        self.layer = nn.Linear(10, 1)  # Example layer

    def forward(self, x):
        rna_data, tissue_id = zip(*x)
        rna_data = torch.stack(rna_data).to(self.device)
        return self.layer(rna_data)
