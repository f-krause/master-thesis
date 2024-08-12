import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.data = torch.rand(100, 10)
        self.targets = torch.rand(100, 1)
    #      TODO do filtering and pre-processing (if necessary) here
    #      TODO allow for data augmentation, by using different folding algorithms?
    #      TODO remove blacklisted Ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_data_loaders(batch_size, num_workers):
    train_dataset = Dataset(train=True)
    val_dataset = Dataset(train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
