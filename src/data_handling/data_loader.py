import os
import torch
import pickle
from omegaconf import OmegaConf, DictConfig
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from torch.utils.data import DataLoader
from log.logger import setup_logger

TOKENS = 'ACGT().BEHIMSX'


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, fold: int, train_val: bool = False, val: bool = False, test: bool = False):
        self.rna_data = None
        self.tissue_ids = None
        self.targets = None

        logger = setup_logger()

        if sum([train_val, val, test]) > 1:
            raise ValueError("Of train_val, val and test only one at a time can be true.")

        if val:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.val_data_file), 'rb') as f:
                logger.info(f"Loading data from: {config.val_data_file}")
                self.rna_data, self.tissue_ids, self.targets = pickle.load(f)
            logger.info(f"Validation dataset with {len(self.rna_data)} samples loaded")
        elif test:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.test_data_file), 'rb') as f:
                logger.info(f"Loading data from: {config.test_data_file}")
                self.rna_data, self.tissue_ids, self.targets = pickle.load(f)
            logger.info(f"Test dataset with {len(self.rna_data)} samples loaded")
        else:
            if config.model == "dummy":
                # Dummy data needed for dev
                logger.warning("USING DUMMY DATA")
                rna_data_full = torch.rand(100, 10)
                tissue_ids_full = np.random.choice(range(10), 100)
                targets_full = torch.rand(100, 1)
            else:
                # Actual train data needed
                with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_train", config.train_data_file),
                          'rb') as f:
                    logger.info(f"Loading data from: {config.train_data_file}")
                    rna_data_full, tissue_ids_full, targets_full = pickle.load(f)  # n x 3, n, n
                    if len(rna_data_full) < 10000:
                        logger.warning(f"DATASET HAS ONLY {len(rna_data_full)} SAMPLES")  # TODO

            train_indices, val_indices = self._get_train_val_indices(rna_data_full, targets_full, fold, config.seed,
                                                                     config.nr_folds)
            if train_val:
                self.rna_data = [rna_data_full[i] for i in val_indices]
                self.tissue_ids = tissue_ids_full[val_indices]
                self.targets = targets_full[val_indices]
                logger.info(f"Train validation dataset with {len(self.rna_data)} samples loaded")
            else:
                self.rna_data = [rna_data_full[i] for i in train_indices]
                self.tissue_ids = tissue_ids_full[train_indices]
                self.targets = targets_full[train_indices]
                logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")

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
        return [self.rna_data[index], self.tissue_ids[index]], self.targets[index]


def _pad_sequences(batch):
    data, targets = zip(*batch)
    rna_data, tissue_ids = zip(*data)
    tissue_ids = torch.tensor(tissue_ids)
    seq_lengths = torch.tensor([seq.size(0) for seq in rna_data])
    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

    return [rna_data_padded, tissue_ids, seq_lengths], torch.tensor(targets)


def get_train_data_loaders(config: DictConfig, fold: int):
    train_dataset = RNADataset(config, fold)
    val_dataset = RNADataset(config, fold, train_val=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers,
                              collate_fn=_pad_sequences)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=_pad_sequences)
    return train_loader, val_loader


def get_val_data_loader(config: DictConfig):
    val_dataset = RNADataset(config, 0, val=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=_pad_sequences)
    return val_loader


def get_test_data_loader(config: DictConfig):
    test_dataset = RNADataset(config, 0, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, collate_fn=_pad_sequences)
    return test_loader


if __name__ == "__main__":
    # for debugging
    from utils import set_project_path, set_log_file

    dev_config = OmegaConf.create(
        {"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
         "train_data_file": "codon_train_2.7k_data.pkl", "val_data_file": "codon_val_2.7k_data.pkl",
         "test_data_file": "codon_test_2.7k_data.pkl", "batch_size": 4, "num_workers": 4,
         "folding_algorithm": "viennarna", "seed": 42, "nr_folds": 5}
    )
    set_project_path(dev_config)
    set_log_file(dev_config)
    # train_dataloader = RNADataset(config, train=True)
    # print(len(train_dataloader))

    print("Testing train and train_val data loaders")
    train_loader_test, train_val_loader_test = get_train_data_loaders(dev_config, fold=1)
    data_iter = iter(train_loader_test)
    x, y = next(data_iter)
    print(x)
    print(y)

    print("Testing val data loader")
    val_loader_test = get_val_data_loader(dev_config)
    data_iter = iter(val_loader_test)
    x, y = next(data_iter)
    print(y)

    print("Testing test data loader")
    test_loader_test = get_test_data_loader(dev_config)
    data_iter = iter(test_loader_test)
    x, y = next(data_iter)
    print(x)
