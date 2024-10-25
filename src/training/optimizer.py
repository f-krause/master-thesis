import torch.optim as optim
from ranger import Ranger
from omegaconf import DictConfig


def get_optimizer(model, optimizer_config: DictConfig):
    optimizer_name = optimizer_config.name

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=optimizer_config.lr)
    elif optimizer_name == 'ranger':
        return Ranger(model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=optimizer_config.lr, momentum=optimizer_config.momentum)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
