# Create train, validation and test data for CODON dataset (regression and binary classification task)

import os
import pickle
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.knowledge_db import CODON_MAP_DNA
from train_val_test_indices import get_train_val_test_indices
from data_utils import store_data, check_identical

MAX_SEQ_LENGTH = 8_100  # Maximum number of NUCLEOTIDES in CDS (note: 3' and 5' tails (UTR) are removed)
MAX_DATA = 300_000  # 182_625 seq-tuple pairs in total
SEED = 1192  # randomly drawn with np.random.randint(0,2024) on 22.10.2024, 15:00


def get_train_data_file(file_name: str, regression=False, check_reproduce=False):
    """Store data for training, validation and testing"""
    print("Loading data")
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    identifiers = []
    sequences = []
    tissue_ids = []
    rna_data = []
    targets = []
    targets_bin = []

    print("Starting data processing")
    for identifier, content in tqdm(raw_data.items()):
        sequence = content['fasta']
        bed_annotation = content['bed_annotation']

        # value: 0, low-PTR: 1, high-PTR: 2
        data_targets_bin = (np.where(np.isnan(content["targets"]), np.nan, 0) +
                            np.where(np.isnan(content["targets_bin"]), 0, content["targets_bin"] + 1))

        coding_sequence = [nucleotide for nucleotide, annotation in zip(list(sequence), bed_annotation) if
                           annotation not in [5, 3]]  # CDS, drop 5' and 3' UTR
        encoded_sequence = [CODON_MAP_DNA.get(''.join(coding_sequence[i:i + 3]), -1) for i in
                            range(0, len(coding_sequence), 3)]

        if len(coding_sequence) > MAX_SEQ_LENGTH:
            continue
        else:
            if regression:
                identifiers.append(identifier)
                sequences.append("".join(coding_sequence))
                rna_data.append(torch.tensor(encoded_sequence))
                targets.append(np.nanmean(content['targets']))

                tissue_ids.append(torch.tensor(0))  # dummy legacy
                targets_bin.append(torch.tensor(0))  # dummy legacy
            else:
                for tissue_id, target in enumerate(content['targets']):
                    if np.isnan(target):
                        continue

                    identifiers.append(identifier)
                    sequences.append("".join(coding_sequence))
                    rna_data.append(torch.tensor(encoded_sequence))
                    tissue_ids.append(tissue_id)
                    targets.append(target)
                    targets_bin.append(int(data_targets_bin[tissue_id]))

                    if len(rna_data) >= MAX_DATA:
                        break
            if len(rna_data) >= MAX_DATA:
                break

    train_indices, val_indices, test_indices = get_train_val_test_indices(sequences, random_state=SEED)

    print("Num seq-tuple pairs TRAIN:", len(train_indices))
    print("Num seq-tuple pairs VAL:", len(val_indices))
    print("Num seq-tuple pairs TEST:", len(test_indices))

    max_seq_len_logging = str(MAX_SEQ_LENGTH / 1000) + "k"

    if check_reproduce:
        check_identical(train_indices, identifiers, tissue_ids,
                        f"data/data_train/{file_name}train_{max_seq_len_logging}")
        check_identical(val_indices, identifiers, tissue_ids,
                        f"data/data_test/{file_name}val_{max_seq_len_logging}")
        check_identical(test_indices, identifiers, tissue_ids,
                        f"data/data_test/{file_name}test_{max_seq_len_logging}")
    else:
        print("Storing data")
        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, train_indices,
                   f"data/data_train/{file_name}train_{max_seq_len_logging}")

        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, val_indices,
                   f"data/data_test/{file_name}val_{max_seq_len_logging}")

        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, test_indices,
                   f"data/data_test/{file_name}test_{max_seq_len_logging}")

        print("Data successfully created")


if __name__ == '__main__':
    from utils.utils import set_project_path

    CHECK_REPRODUCTION = False
    MEAN_REGRESSION = False  # create regression dataset with mean target across tissues
    FILE_NAME = "codon"

    if FILE_NAME:
        FILE_NAME += "_"

    dev_config = OmegaConf.create(
        {"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline", "pretrain": False,
         "batch_size": 32, "num_workers": 4, "train_data_file": "", "dev": True, "binary_class": False,
         "frequency_features": False})
    set_project_path(dev_config)

    get_train_data_file(FILE_NAME, regression=MEAN_REGRESSION, check_reproduce=CHECK_REPRODUCTION)
