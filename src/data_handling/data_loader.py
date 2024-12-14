import os
import torch
import pickle
from omegaconf import OmegaConf, DictConfig
import numpy as np
from torch.utils.data import DataLoader
from log.logger import setup_logger
from itertools import compress

from sklearn.model_selection import train_test_split
from data_handling.train_val_test_indices import get_train_val_test_indices
from data_handling.data_utils import fit_evaluate_simple_models
from knowledge_db import CODON_MAP_DNA, TISSUES


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, fold: int, train_val: bool = False, val: bool = False, test: bool = False):
        self.rna_data = None
        self.tissue_ids = None
        self.targets = None

        logger = setup_logger()

        if sum([train_val, val, test]) > 1:
            raise ValueError("Of train_val, val and test only one at a time can be true.")

        if val:
            with (open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.val_data_file), 'rb') as f):
                logger.info(f"Loading data from: {config.val_data_file}")
                rna_data, tissue_ids, targets, targets_bin = pickle.load(f)
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = \
                    self.filter_data(config, rna_data, tissue_ids, targets, targets_bin, logger)
            logger.info(f"Validation dataset with {len(self.rna_data)} samples loaded")
        elif test:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.test_data_file), 'rb') as f:
                rna_data, tissue_ids, targets, targets_bin = pickle.load(f)
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = \
                    self.filter_data(config, rna_data, tissue_ids, targets, targets_bin, logger)
            logger.info(f"Test dataset with {len(self.rna_data)} samples loaded")
        else:
            # Actual train data needed
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_train", config.train_data_file),
                      'rb') as f:
                logger.info(f"Loading data from: {config.train_data_file}")
                rna_data_full, tissue_ids_full, targets_full, targets_bin_full = pickle.load(f)  # n x 3, n, n
                if len(rna_data_full) < 10000:
                    logger.warning(f"DATASET HAS ONLY {len(rna_data_full)} SAMPLES")

                rna_data_full, tissue_ids_full, targets_full, targets_bin_full = \
                    self.filter_data(config, rna_data_full, tissue_ids_full, targets_full, targets_bin_full, logger)

            if config.val_fraction_of_train > 0:
                inverted_codon_map = {value: key for key, value in CODON_MAP_DNA.items()}
                rna_data_full_inverted = [[inverted_codon_map[int(idx)] for idx in rna_data] for rna_data in
                                          rna_data_full]
                mrna_sequences = ["".join(map(str, seq)) for seq in rna_data_full_inverted]
                # mrna_sequences = ["".join(map(str, tensor.tolist())) for tensor in
                #                   rna_data_full]  # legacy: to reproduce optuna training
                train_indices, val_indices = self._get_train_val_indices(config, mrna_sequences, fold)

                logger.info(
                    f"Distribution seq lens - full data: {np.histogram([len(seq) for seq in mrna_sequences], bins=10)}")

                if train_val:
                    self.rna_data = [rna_data_full[i] for i in val_indices]
                    self.tissue_ids = tissue_ids_full[val_indices]
                    self.targets = targets_full[val_indices]
                    self.targets_bin = targets_bin_full[val_indices]

                    # SANITY CHECK DISTRIBUTION
                    mrna_sequences = [mrna_sequences[i] for i in val_indices]
                    logger.info(
                        f"Distribution seq lens - train val set: {np.histogram([len(seq) for seq in mrna_sequences], bins=10)}")

                    logger.info(f"Train validation dataset with {len(self.rna_data)} samples loaded")
                else:
                    self.rna_data = [rna_data_full[i] for i in train_indices]
                    self.tissue_ids = tissue_ids_full[train_indices]
                    self.targets = targets_full[train_indices]
                    self.targets_bin = targets_bin_full[train_indices]

                    # SANITY CHECK DISTRIBUTION
                    mrna_sequences = [mrna_sequences[i] for i in train_indices]
                    logger.info(
                        f"Distribution seq lens - train set: {np.histogram([len(seq) for seq in mrna_sequences], bins=10)}")

                    logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")
            else:
                self.rna_data = rna_data_full
                self.tissue_ids = tissue_ids_full
                self.targets = targets_full
                self.targets_bin = targets_bin_full

                logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")

        # cast from int8 to int for computations
        self.rna_data = [rna_data.int() for rna_data in self.rna_data]
        self.tissue_ids = self.tissue_ids.int()
        self.targets_bin = self.targets_bin.int()

    def _get_train_val_indices(self, config: DictConfig, mrna_sequences, fold):
        if config.nr_folds == 1:
            # train_indices, val_indices = train_test_split(range(len(mrna_sequences)), test_size=0.2,
            #                                               random_state=random_state)  # legacy
            train_indices, val_indices, _ = get_train_val_test_indices(mrna_sequences,
                                                                       val_frac=config.val_fraction_of_train,
                                                                       test_frac=0, random_state=config.seed)

        elif config.nr_folds == 3:
            train_indices, val_indices = self._get_3_fold_indices(mrna_sequences, fold, random_state=config.seed)
        else:
            raise ValueError("Only 1 and 3 folds are currently supported")
        return train_indices, val_indices

    @staticmethod
    def _get_3_fold_indices(mrna_sequences, fold, random_state=42):
        indices_triple = get_train_val_test_indices(mrna_sequences, val_frac=0.33, test_frac=0.33,
                                                    random_state=random_state)
        indices_triple = list(indices_triple)
        val_indices = indices_triple.pop(fold)
        train_indices = np.concatenate(indices_triple)

        return train_indices.tolist(), val_indices

    @staticmethod
    def filter_data(config, rna_data, tissue_ids, targets, targets_bin, logger):
        mask = torch.ones((len(rna_data)), dtype=torch.bool)

        if config.tissue_id in range(len(TISSUES)):
            mask = tissue_ids == config.tissue_id
            logger.warning(f"Only keeping data for tissue {TISSUES[config.tissue_id]}")

        if config.binary_class:
            mask_bin = targets_bin > 0  # only keep low-/high-PTR samples
            mask = mask_bin & mask
            targets_bin -= 1  # make binary class 0/1 encoded
            logger.warning("Only keeping data for binary CLASSIFICATION")

        rna_data = list(compress(rna_data, mask))
        return [rna_data] + [tensor[mask] for tensor in [tissue_ids, targets, targets_bin]]

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, index):
        return [self.rna_data[index], self.tissue_ids[index]], self.targets[index], self.targets_bin[index]


def _pad_sequences(batch):
    data, targets, targets_bin = zip(*batch)
    rna_data, tissue_ids = zip(*data)
    tissue_ids = torch.tensor(tissue_ids)
    seq_lengths = torch.tensor([seq.size(0) for seq in rna_data])

    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

    return [rna_data_padded, tissue_ids, seq_lengths], torch.tensor(targets), torch.tensor(targets_bin)


def _pad_sequences_and_reverse(batch):
    data, targets, targets_bin = zip(*batch)
    rna_data, tissue_ids = zip(*data)
    tissue_ids = torch.tensor(tissue_ids)
    seq_lengths = torch.tensor([seq.size(0) for seq in rna_data])

    rna_data = [torch.flip(seq, dims=[0]) if torch.rand(1).item() < 0.5 else seq for seq in rna_data]

    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

    return [rna_data_padded, tissue_ids, seq_lengths], torch.tensor(targets), torch.tensor(targets_bin)


def get_train_data_loaders(config: DictConfig, fold: int):
    train_dataset = RNADataset(config, fold)
    if config.val_fraction_of_train > 0:
        val_dataset = RNADataset(config, fold, train_val=True)
    else:
        val_dataset = RNADataset(config, fold, val=True)

    # fit_evaluate_simple_models(train_dataset, val_dataset, config.binary_class)

    if config.random_reverse:
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, collate_fn=_pad_sequences_and_reverse)
        val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, collate_fn=_pad_sequences_and_reverse)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, collate_fn=_pad_sequences)
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
        {"project_path": None, "log_file_path": None, "subproject": "dev/delete_me", "dev": True, "model": "baseline",
         "train_data_file": "codon_train_2.7k_data.pkl", "val_data_file": "codon_val_2.7k_data.pkl",
         "test_data_file": "codon_test_2.7k_data.pkl", "batch_size": 4, "num_workers": 4, "val_fraction_of_train": 0.15,
         "folding_algorithm": "viennarna", "seed": 42, "nr_folds": 3, "random_reverse": True,
         "tissue_id": -1, "binary_class": True, "frequency_features": False}
    )
    set_project_path(dev_config)
    set_log_file(dev_config)

    print("Testing train and train_val data loaders")
    train_loader_test, train_val_loader_test = get_train_data_loaders(dev_config, fold=0)
    data_iter = iter(train_loader_test)
    x, y, y2 = next(data_iter)
    print(len(x[0]))
    print(y, y2)

    print("Testing val data loader")
    val_loader_test = get_val_data_loader(dev_config)
    data_iter = iter(val_loader_test)
    x, _, _ = next(data_iter)
    print(len(x[0]))

    print("Testing test data loader")
    test_loader_test = get_test_data_loader(dev_config)
    data_iter = iter(test_loader_test)
    x, _, _ = next(data_iter)
    print(len(x[0]))
