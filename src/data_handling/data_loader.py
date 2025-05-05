import os
import torch
import pickle
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from log.logger import setup_logger
from itertools import compress

from data_handling.train_val_test_indices import get_train_val_test_indices
from data_handling.data_utils import cv_simple_models, train_validate_simple_model
from utils.knowledge_db import CODON_MAP_DNA, TISSUES, TOKENS
# from sklearn.model_selection import train_test_split


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, config: DictConfig, fold: int, train_val: bool = False, val: bool = False, test: bool = False):
        self.config = config
        self.rna_data = None
        self.tissue_ids = None
        self.targets = None

        self.logger = setup_logger()

        if sum([train_val, val, test]) > 1:
            raise ValueError("Of train_val, val and test only one at a time can be true.")

        if val:
            self.logger.info(f"Loading val data from: {config.val_data_file}")
            with (open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.val_data_file), 'rb') as f):
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = pickle.load(f)
            self.rna_data = self._set_codon_or_nucl_setting(self.rna_data)
            self.rna_data, self.tissue_ids, self.targets, self.targets_bin = \
                self._filter_data(self.rna_data, self.tissue_ids, self.targets, self.targets_bin)

            self.logger.info(f"Validation dataset with {len(self.rna_data)} samples loaded")
            self._get_dataset_stats()

        elif test:
            self.logger.info(f"Loading test data from: {config.test_data_file}")
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.test_data_file), 'rb') as f:
                self.rna_data, self.tissue_ids, self.targets, self.targets_bin = pickle.load(f)
            self.rna_data = self._set_codon_or_nucl_setting(self.rna_data)
            self.rna_data, self.tissue_ids, self.targets, self.targets_bin = \
                self._filter_data(self.rna_data, self.tissue_ids, self.targets, self.targets_bin)

            self.logger.info(f"Test dataset with {len(self.rna_data)} samples loaded")
            self._get_dataset_stats()

        else:
            # Actual train data needed
            with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_train", config.train_data_file),
                      'rb') as f:
                self.logger.info(f"Loading train data from: {config.train_data_file}")
                rna_data_full, tissue_ids_full, targets_full, targets_bin_full = pickle.load(f)  # n x 3, n, n
            rna_data_full = self._set_codon_or_nucl_setting(rna_data_full)

            rna_data_full, tissue_ids_full, targets_full, targets_bin_full = \
                self._filter_data(rna_data_full, tissue_ids_full, targets_full, targets_bin_full)

            if config.concat_train_val:
                self.logger.info(f"Loading val data from: {config.val_data_file}")
                with (open(os.path.join(os.environ["PROJECT_PATH"], "data/data_test", config.val_data_file),
                           'rb') as f):
                    val_rna_data, val_tissue_ids, val_targets, val_targets_bin = pickle.load(f)
                val_rna_data = self._set_codon_or_nucl_setting(val_rna_data)
                val_rna_data, val_tissue_ids, val_targets, val_targets_bin = \
                    self._filter_data(val_rna_data, val_tissue_ids, val_targets, val_targets_bin)

                rna_data_full += val_rna_data
                tissue_ids_full = torch.cat((tissue_ids_full, val_tissue_ids))
                targets_full = torch.cat((targets_full, val_targets))
                targets_bin_full = torch.cat((targets_bin_full, val_targets_bin))

                self.logger.info(f"Adding validation dataset with {len(val_rna_data)} samples to train")

            if config.nr_folds > 1:
                # Run Cross validation
                inverted_codon_map = {value: key for key, value in CODON_MAP_DNA.items()}
                rna_data_full_inverted = [[inverted_codon_map[int(idx)] for idx in rna_data] for rna_data in
                                          rna_data_full]
                mrna_sequences = ["".join(map(str, seq)) for seq in rna_data_full_inverted]
                # mrna_sequences = ["".join(map(str, tensor.tolist())) for tensor in
                #                   rna_data_full]  # legacy: to reproduce optuna training
                train_indices, val_indices = self._get_train_val_indices(config, mrna_sequences, fold)

                self.logger.info(
                    f"Distribution seq lens - full data: {np.histogram([len(seq) for seq in mrna_sequences], bins=10)}")

                if train_val:
                    self.rna_data = [rna_data_full[i] for i in val_indices]
                    self.tissue_ids = tissue_ids_full[val_indices]
                    self.targets = targets_full[val_indices]
                    self.targets_bin = targets_bin_full[val_indices]

                    self.logger.info(f"Train validation dataset with {len(self.rna_data)} samples loaded")
                    self._get_dataset_stats()

                else:
                    self.rna_data = [rna_data_full[i] for i in train_indices]
                    self.tissue_ids = tissue_ids_full[train_indices]
                    self.targets = targets_full[train_indices]
                    self.targets_bin = targets_bin_full[train_indices]

                    self.logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")
                    self._get_dataset_stats()

            else:
                # No cross validation, just load the full data
                self.rna_data = rna_data_full
                self.tissue_ids = tissue_ids_full
                self.targets = targets_full
                self.targets_bin = targets_bin_full

                self.logger.info(f"Train dataset with {len(self.rna_data)} samples loaded")
                self._get_dataset_stats()

            if len(rna_data_full) < 10000:
                self.logger.warning(f"TRAIN DATASET HAS ONLY {len(rna_data_full)} SAMPLES")

        # cast from int8 to int for computations
        self.rna_data = [r.int() for r in self.rna_data]  # TODO isn't this only necessary for nucleotide data?
        self.tissue_ids = self.tissue_ids.int()
        self.targets_bin = self.targets_bin.int()

        # adding frequencies to rna_data
        if config.model.lower() == "ptrnet":
            self.freqs = self._compute_frequencies_nucleotides(self.rna_data)
        else:
            self.freqs = self._compute_frequencies_codons(self.rna_data)

    @staticmethod
    def _compute_frequencies_nucleotides(rna_data_nucleotides):
        freqs = []
        for rna in rna_data_nucleotides:
            rna = rna.permute(1, 0)[0].tolist()
            rna_decoded = [TOKENS[i - 1] for i in rna]
            codon_seq = [CODON_MAP_DNA.get(''.join(rna_decoded[i:i + 3]), -1) for i in
                         range(0, len(rna_decoded), 3)]
            codon_seq = torch.tensor(codon_seq)
            codon_seq = codon_seq[codon_seq != -1]
            counts = torch.bincount(torch.tensor(codon_seq), minlength=len(CODON_MAP_DNA) + 1)[1:len(CODON_MAP_DNA) + 1]
            freq = counts.float() / counts.sum()
            freqs.append(freq)
        return torch.stack(freqs)

    @staticmethod
    def _compute_frequencies_codons(rna_data_codons):
        freqs = []
        for rna in rna_data_codons:
            # codon_seq = [CODON_MAP_DNA.get(''.join(rna_decoded[i:i + 3]), -1) for i in
            #                     range(0, len(rna_decoded), 3)]
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA) + 1)[1:len(CODON_MAP_DNA) + 1]
            freq = counts.float() / counts.sum()
            freqs.append(freq)
        return torch.stack(freqs)

    def _get_train_val_indices(self, config: DictConfig, mrna_sequences, fold):
        if config.nr_folds == 3:
            train_indices, val_indices = self._get_3_fold_indices(mrna_sequences, fold, random_state=config.seed)
        else:
            raise ValueError("Only 3 folds are currently supported for cross validation.")
        return train_indices, val_indices

    @staticmethod
    def _get_3_fold_indices(mrna_sequences, fold, random_state=42):
        indices_triple = get_train_val_test_indices(mrna_sequences, val_frac=0.33, test_frac=0.33,
                                                    random_state=random_state)
        indices_triple = list(indices_triple)
        val_indices = indices_triple.pop(fold)
        train_indices = np.concatenate(indices_triple)

        return train_indices.tolist(), val_indices

    def _filter_data(self, rna_data, tissue_ids, targets, targets_bin):
        mask = torch.ones((len(rna_data)), dtype=torch.bool)

        if self.config.tissue_id in range(len(TISSUES)):
            mask = tissue_ids == self.config.tissue_id
            self.logger.warning(f"Only keeping data for TISSUE {TISSUES[self.config.tissue_id]}")

        if self.config.binary_class:
            mask_bin = targets_bin > 0  # only keep low-/high-PTR samples
            mask = mask_bin & mask
            targets_bin -= 1  # make binary class 0/1 encoded
            self.logger.warning("Only keeping data for binary CLASSIFICATION")

        if "binary_class" in self.config.train_data_file:
            mask_len = torch.tensor([len(d) <= self.config.max_seq_length for d in rna_data])
            mask = mask_len & mask

        rna_data = list(compress(rna_data, mask))

        return [rna_data] + [tensor[mask] for tensor in [tissue_ids, targets, targets_bin]]

    def _set_codon_or_nucl_setting(self, rna_data):
        if "binary_class" in self.config.train_data_file:
            if self.config.model.lower() == "ptrnet":
                rna_data = [d["nucleotide_rna_data"] for d in rna_data]
            else:
                rna_data = [d["codon_rna_data"] for d in rna_data]
        return rna_data

    def _get_dataset_stats(self):
        if self.config.binary_class:
            df = pd.DataFrame({"tissue_id": self.tissue_ids.tolist(), "targets_bin": self.targets_bin.tolist(),
                               "targets": self.targets.tolist()})
            self.logger.info("# Stats of dataset after filtering")
            self.logger.info(f"  Class distribution:\n{df.groupby(['tissue_id', 'targets_bin']).targets.count()}")
        else:
            pass  # could implement regression setting specific stats here
        self.logger.info(f"  Seq len distribution:\n{np.histogram([len(seq) for seq in self.rna_data], bins=10)}")

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, index):
        return ([self.rna_data[index], self.tissue_ids[index], self.freqs[index]], self.targets[index],
                self.targets_bin[index])


def _pad_sequences(batch):
    data, targets, targets_bin = zip(*batch)
    rna_data, tissue_ids, freqs = zip(*data)
    freqs = torch.stack(freqs)
    tissue_ids = torch.tensor(tissue_ids)
    seq_lengths = torch.tensor([seq.size(0) for seq in rna_data])

    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

    return [rna_data_padded, tissue_ids, seq_lengths, freqs], torch.tensor(targets), torch.tensor(targets_bin)


def _pad_sequences_and_reverse(batch):
    data, targets, targets_bin = zip(*batch)
    rna_data, tissue_ids, freqs = zip(*data)
    freqs = torch.stack(freqs)
    tissue_ids = torch.tensor(tissue_ids)
    seq_lengths = torch.tensor([seq.size(0) for seq in rna_data])

    rna_data = [torch.flip(seq, dims=[0]) if torch.rand(1).item() < 0.5 else seq for seq in rna_data]

    rna_data_padded = torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True)

    return [rna_data_padded, tissue_ids, seq_lengths, freqs], torch.tensor(targets), torch.tensor(targets_bin)


def get_train_data_loaders(config: DictConfig, fold: int):
    train_dataset = RNADataset(config, fold)
    if config.nr_folds > 1:
        # Running cross validation
        val_dataset = RNADataset(config, fold, train_val=True)
    elif config.evaluate_on_test:
        val_dataset = RNADataset(config, fold, test=True)
    else:
        val_dataset = RNADataset(config, fold, val=True)

    # cv_simple_models(train_dataset, val_dataset, config.binary_class)
    # train_validate_simple_model(train_dataset, val_dataset, config.binary_class)

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
    from utils.utils import set_project_path, set_log_file, check_config

    dev_config = OmegaConf.create(
        {
            "project_path": None, "log_file_path": None, "subproject": "dev/delete_me", "dev": True,

            "model": "ptrnet", "train_data_file": "dev_train_9.0k_data.pkl", "val_data_file": "dev_val_9.0k_data.pkl",
            "test_data_file": "dev_test_9.0k_data.pkl",  # test nucleotide level data

            # "model": "baseline", "train_data_file": "dev_codon_train_8.1k_data.pkl",
            # "val_data_file": "dev_codon_val_8.1k_data.pkl", "test_data_file": "dev_codon_test_8.1k_data.pkl", # test codon level data

            "batch_size": 4, "num_workers": 4, "folding_algorithm": "viennarna", "seed": 42,
            "nr_folds": 0,
            "random_reverse": True,
            "tissue_id": -1,
            "binary_class": True,
            "frequency_features": True,
            "concat_train_val": True,
            "evaluate_on_test": True
         }
    )
    check_config(dev_config)
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
