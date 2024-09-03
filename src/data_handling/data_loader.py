import os
import torch
import pickle
from box import Box
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch.utils.data import DataLoader
from log.logger import setup_logger

tokens = 'ACGT().BEHIMSX'


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, config: Box, fold: int, train: bool = True):
        self.data = None
        self.targets = None

        logger = setup_logger()

        if config.model == "dummy":
            logger.warning("USING DUMMY DATA")
            data_full = torch.rand(100, 10)
            targets_full = torch.rand(100, 1)
        else:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/dev_train_data_small.pkl"), 'rb') as f:  # TODO
                logger.warning("LOADING SMALL DEV TRAINING DATA")
                data_full, targets_full = pickle.load(f)
                data_full = np.array(data_full)
                targets_full = torch.tensor(targets_full, dtype=torch.float64)

        train_indices, val_indices = self._get_train_val_indices(data_full, targets_full, fold, config.seed,
                                                                 config.nr_folds)

        if train:
            self.data = data_full[train_indices]
            self.targets = targets_full[train_indices]
            logger.info(f"Train dataset with {len(self.data)} samples loaded")
        else:
            self.data = data_full[val_indices]
            self.targets = targets_full[val_indices]
            logger.info(f"Validation dataset with {len(self.data)} samples loaded")

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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def get_data_loaders(config: Box, fold: int):
    train_dataset = RNADataset(config, fold, train=True)
    val_dataset = RNADataset(config, fold, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    # for debugging
    from utils import set_project_path, set_log_file

    config = Box({"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
                  "batch_size": 32, "num_workers": 4, "folding_algorithm": "viennarna", "seed": 42, "nr_folds": 5})
    set_project_path(config)
    set_log_file(config)
    # train_dataloader = RNADataset(config, train=True)
    # print(len(train_dataloader))
    train_loader, val_loader = get_data_loaders(config, 1)
    print(len(train_loader), len(val_loader))
