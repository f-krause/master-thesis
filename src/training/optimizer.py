import torch.optim as optim
from omegaconf import OmegaConf


def get_optimizer(model, optimizer_config: OmegaConf):
    optimizer_name = optimizer_config.name

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=optimizer_config.lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=optimizer_config.lr, momentum=optimizer_config.momentum)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
