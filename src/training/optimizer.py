import torch.optim as optim
from box import Box


def get_optimizer(model, optimizer_config: Box):
    optimizer_name = optimizer_config.name

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=optimizer_config.lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=optimizer_config.lr, momentum=optimizer_config.momentum)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
