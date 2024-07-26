import torch.optim as optim


def get_optimizer(model, config):
    optimizer_name = config['optimizer']['name']
    lr = config['optimizer']['lr']

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=config['optimizer']['momentum'])
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
