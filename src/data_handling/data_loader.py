import torch
from torch.utils.data import DataLoader
from utils import TrainConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config: TrainConfig, train=True, dummy=False):
        if config.model == "dummy":
            self.data = torch.rand(100, 10)
            self.targets = torch.rand(100, 1)
        else:
            # TODO
            raise NotImplementedError()

    #      TODO do filtering and pre-processing (if necessary) here
    #      TODO allow for data augmentation, by using different folding algorithms?
    #      TODO remove blacklisted Ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_data_loaders(config: TrainConfig):
    train_dataset = Dataset(config, train=True)
    val_dataset = Dataset(config, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

    return train_loader, val_loader
