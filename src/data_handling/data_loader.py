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
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = pickle.load(f)
            logger.info(f"Validation dataset with {len(self.rna_data)} samples loaded")
        elif test:
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.test_data_file), 'rb') as f:
                logger.info(f"Loading data from: {config.test_data_file}")
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = pickle.load(f)
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
                    rna_data_full, tissue_ids_full, targets_full, targets_bin_full = pickle.load(f)  # n x 3, n, n
                    if len(rna_data_full) < 10000:
                        logger.warning(f"DATASET HAS ONLY {len(rna_data_full)} SAMPLES")

            mask = torch.ones((len(rna_data_full)), dtype=torch.bool)

            if config.tissue_id in range(len(TISSUES)):
                mask = tissue_ids_full == config.tissue_id
                logger.warning(f"Only keeping data for tissue {TISSUES[config.tissue_id]}")

            if config.binary_class:
                mask_bin = targets_bin_full > 0  # only keep low-/high-PTR samples
                mask = mask_bin & mask
                targets_bin_full -= 1  # make binary class 0/1 encoded
                logger.warning("Only keeping data for binary CLASSIFICATION")

            rna_data_full = list(compress(rna_data_full, mask))
            tissue_ids_full, targets_full, targets_bin_full = \
                [tensor[mask] for tensor in [tissue_ids_full, targets_full, targets_bin_full]]

            mrna_sequences = ["".join(map(str, tensor.tolist())) for tensor in
                              rna_data_full]  # FIXME: mrna sequence length is now not fully identical with the original sequence length
            train_indices, val_indices = self._get_train_val_indices(mrna_sequences, fold, config.seed, config.nr_folds)

            if train_val:
                self.rna_data = [rna_data_full[i] for i in val_indices]
                self.tissue_ids = tissue_ids_full[val_indices]
                self.targets = targets_full[val_indices]
                self.targets_bin = targets_bin_full[val_indices]

                # FIXME
                # only keep data for tissue_id == 12
                # tissue_id_12_indices = np.where(self.tissue_ids == 12)[0]
                # self.rna_data = [self.rna_data[i] for i in tissue_id_12_indices]
                # self.tissue_ids = self.tissue_ids[tissue_id_12_indices]
                # self.targets = self.targets[tissue_id_12_indices]
                # logger.warning("DEV: Only keeping data for tissue_id == 12")

                logger.info(f"Train validation dataset with {len(self.rna_data)} samples loaded")
            else:
                self.rna_data = [rna_data_full[i] for i in train_indices]
                self.tissue_ids = tissue_ids_full[train_indices]
                self.targets = targets_full[train_indices]
                self.targets_bin = targets_bin_full[train_indices]

                # FIXME
                # only keep data for tissue_id == 12
                # tissue_id_12_indices = np.where(self.tissue_ids == 12)[0]
                # self.rna_data = [self.rna_data[i] for i in tissue_id_12_indices]
                # self.tissue_ids = self.tissue_ids[tissue_id_12_indices]
                # self.targets = self.targets[tissue_id_12_indices]
                # logger.warning("DEV: Only keeping data for tissue_id == 12")

                logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")

    def _get_train_val_indices(self, mrna_sequences, fold, random_state=42, nr_folds=3):
        if nr_folds == 1:
            # train_indices, val_indices = train_test_split(range(len(mrna_sequences)), test_size=0.2,
            #                                               random_state=random_state)  # legacy
            train_indices, val_indices, _ = get_train_val_test_indices(mrna_sequences, val_frac=0.15, test_frac=0,
                                                                       random_state=random_state)
        elif nr_folds == 3:
            train_indices, val_indices = self._get_3_fold_indices(mrna_sequences, fold, random_state=random_state)
        else:
            raise ValueError("Only 1 and 3 folds are currently supported")
        return train_indices, val_indices

    @staticmethod
    def _get_3_fold_indices(mrna_sequences, fold, random_state=42):
        # TODO test this!
        indices_triple = get_train_val_test_indices(mrna_sequences, val_frac=0.33, test_frac=0.33,
                                                    random_state=random_state)

        val_indices = indices_triple.pop(fold)
        train_indices = np.concatenate(indices_triple)

        return train_indices.tolist(), val_indices

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


def get_train_data_loaders(config: DictConfig, fold: int):
    train_dataset = RNADataset(config, fold)
    val_dataset = RNADataset(config, fold, train_val=True)

    fit_evaluate_simple_models(train_dataset, val_dataset)

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
        {"project_path": None, "log_file_path": None, "subproject": "dev/delete_me", "model": "baseline",
         "train_data_file": "bin_codon_train_2.7k_data.pkl", "val_data_file": "bin_codon_val_2.7k_data.pkl",
         "test_data_file": "bin_codon_test_2.7k_data.pkl", "batch_size": 4, "num_workers": 4,
         "folding_algorithm": "viennarna", "seed": 42, "nr_folds": 1,
         "tissue_id": 28,
         "binary_class": False}
    )
    set_project_path(dev_config)
    set_log_file(dev_config)

    print("Testing train and train_val data loaders")
    train_loader_test, train_val_loader_test = get_train_data_loaders(dev_config, fold=1)
    data_iter = iter(train_loader_test)
    x, y, y2 = next(data_iter)
    print(x.size())
    print(y, y2)

    print("Testing val data loader")
    val_loader_test = get_val_data_loader(dev_config)
    data_iter = iter(val_loader_test)
    x, _, _ = next(data_iter)
    print(x.size())

    print("Testing test data loader")
    test_loader_test = get_test_data_loader(dev_config)
    data_iter = iter(test_loader_test)
    x, _, _ = next(data_iter)
    print(x.size())
