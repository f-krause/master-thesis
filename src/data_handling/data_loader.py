import os
import torch
import pickle
from box import Box
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch.utils.data import DataLoader
from log.logger import setup_logger

TOKENS = 'ACGT().BEHIMSX'


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, config: Box, fold: int, train: bool = True):
        self.rna_data = None
        self.tissue_ids = None
        self.targets = None

        logger = setup_logger()

        if config.model == "dummy":
            logger.warning("USING DUMMY DATA")
            rna_data_full = torch.rand(100, 10)
            targets_full = torch.rand(100, 1)
        else:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/dev_train_data_small.pkl"), 'rb') as f:  # TODO
                logger.warning("LOADING SMALL DEV TRAINING DATA")
                rna_data_full, tissue_ids_full, targets_full = pickle.load(f)  # n x 3, n, n
                targets_full = torch.tensor(targets_full, dtype=torch.float64)

        train_indices, val_indices = self._get_train_val_indices(rna_data_full, targets_full, fold, config.seed,
                                                                 config.nr_folds)

        if train:
            self.rna_data = [rna_data_full[i] for i in train_indices]
            self.tissue_ids = tissue_ids_full[train_indices]
            self.targets = targets_full[train_indices]
            logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")
        else:
            self.rna_data = [rna_data_full[i] for i in val_indices]
            self.tissue_ids = tissue_ids_full[val_indices]
            self.targets = targets_full[val_indices]
            logger.info(f"Validation dataset with {len(self.rna_data)} samples loaded")

    @staticmethod
    def _get_train_val_indices(X, y, fold, seed=42, nr_folds=5):
        if nr_folds == 1:
            return train_test_split(range(len(X)), test_size=0.2, random_state=seed)
        else:
            # splits = StratifiedKFold(n_splits=nr_folds, random_state=seed, shuffle=True)
            splits = KFold(n_splits=nr_folds, random_state=seed, shuffle=True)
            splits = list(splits.split(X))
            train_indices = splits[fold][0]
            val_indices = splits[fold][1]
            return train_indices.tolist(), val_indices.tolist()

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, index):
        return (self.rna_data[index], self.tissue_ids[index]), self.targets[index]


def _pad_sequences(batch):
    # TODO might be useful to pad sequences: pack_padded_sequence
    data, targets = zip(*batch)
    rna_data, tissue_ids = zip(*data)
    lengths = torch.tensor([seq.size(0) for seq in rna_data])
    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)
    # rna_data_packed = torch.nn.utils.rnn.pack_padded_sequence(rna_data_padded, lengths, batch_first=True,
    #                                                           enforce_sorted=False)  # TODO can be useful for LSTM!

    return list(zip(rna_data_padded, tissue_ids)), torch.tensor(targets)


def get_data_loaders(config: Box, fold: int):
    train_dataset = RNADataset(config, fold, train=True)
    val_dataset = RNADataset(config, fold, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers,
                              collate_fn=_pad_sequences)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=_pad_sequences)
    return train_loader, val_loader


if __name__ == "__main__":
    # for debugging
    from utils import set_project_path, set_log_file

    dev_config = Box({"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
                  "batch_size": 32, "num_workers": 4, "folding_algorithm": "viennarna", "seed": 42, "nr_folds": 5})
    set_project_path(dev_config)
    set_log_file(dev_config)
    # train_dataloader = RNADataset(config, train=True)
    # print(len(train_dataloader))
    train_loader, val_loader = get_data_loaders(dev_config, 1)
    data_iter = iter(train_loader)
    x, z = next(data_iter)
    print(x)
    print(len(train_loader), len(val_loader))
