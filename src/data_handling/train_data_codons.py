import os
import json
import pickle

import pandas as pd
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

from knowledge_db import CODON_MAP_DNA
from train_val_test_indices import get_train_val_test_indices

MAX_SEQ_LENGTH = 2700  # Maximum number of codons in CDS (note: 3' and 5' tails (UTR) are removed)
MAX_DATA = 300_000  # 182_625 seq-tuple pairs in total
SEED = 1192  # randomly drawn with np.random.randint(0,2024) on 22.10.2024, 15:00


def _store_data(identifiers: list, rna_data: list, target_ids: list, targets: list, indices: list, path: str):
    identifiers_selected = [identifiers[i] for i in indices]
    rna_data_selected = [rna_data[i] for i in indices]
    target_ids_selected = [target_ids[i] for i in indices]
    targets_selected = [targets[i] for i in indices]
    with open(os.path.join(os.environ["PROJECT_PATH"], path + "_data.pkl"), 'wb') as f:
        pickle.dump([rna_data_selected, torch.tensor(target_ids_selected), torch.tensor(targets_selected)], f)
    pd.DataFrame({"identifier": identifiers_selected, "target_id": target_ids_selected, "index": indices}).to_csv(
        os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"), index=False)


def _check_identical(indices: list, identifiers: list, target_ids: list, path: str):
    """Check reproducibility of data split"""
    identifiers_selected = [identifiers[i] for i in indices]
    target_ids_selected = [target_ids[i] for i in indices]

    selected_set = set(zip(identifiers_selected, target_ids_selected))
    try:
        persistence = pd.read_csv(os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"))
        persistence_set = set(zip(persistence["identifier"].tolist(), persistence["target_id"].tolist()))
        if selected_set != persistence_set:
            raise Exception("REPRODUCTION ISSUE DETECTED")
        else:
            print("Reproducibility check passed!")
    except FileNotFoundError:
        print("Warning: No persistence file found")


def get_train_data_file(file_name: str, check_reproduce=False, return_dict=False):
    """Store data for training, validation and testing"""
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    identifiers = []
    rna_data = []
    targets = []
    target_ids = []
    sequences = []

    for identifier, content in raw_data.items():
        sequence = content['fasta']

        bed_annotation = content['bed_annotation']
        coding_sequence = [nucleotide for nucleotide, annotation in zip(list(sequence), bed_annotation) if
                           annotation not in [5, 3]]  # CDS, drop 5' and 3' UTR
        encoded_sequence = [CODON_MAP_DNA.get(''.join(coding_sequence[i:i + 3]), -1) for i in
                            range(0, len(coding_sequence), 3)]

        if len(coding_sequence) > MAX_SEQ_LENGTH:
            continue
        else:
            for target_id, target in enumerate(content['targets']):
                if np.isnan(target):
                    continue

                sequences.append("".join(coding_sequence))
                identifiers.append(identifier)
                numeric_data = {
                    'codons': encoded_sequence,
                    'target_id': target_id,
                }

                if return_dict:
                    rna_data.append(numeric_data)
                    targets.append(target)
                else:
                    rna_data.append(torch.tensor(numeric_data["codons"]))
                    target_ids.append(target_id)
                    targets.append(target)

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
        _check_identical(train_indices, identifiers, target_ids, f"data/data_train/{file_name}_train_{max_seq_len_logging}")
        _check_identical(val_indices, identifiers, target_ids, f"data/data_test/{file_name}_val_{max_seq_len_logging}")
        _check_identical(test_indices, identifiers, target_ids, f"data/data_test/{file_name}_test_{max_seq_len_logging}")
    else:
        _store_data(identifiers, rna_data, target_ids, targets, train_indices,
                    f"data/data_train/{file_name}_train_{max_seq_len_logging}")

        _store_data(identifiers, rna_data, target_ids, targets, val_indices,
                    f"data/data_test/{file_name}_val_{max_seq_len_logging}")

        _store_data(identifiers, rna_data, target_ids, targets, test_indices,
                    f"data/data_test/{file_name}_test_{max_seq_len_logging}")

        print("Data successfully created")


def _get_structure_pred(identifier: str, config: DictConfig):
    try:
        with open(os.path.join(os.environ["PROJECT_PATH"],
                               f"data/sec_struc/{identifier}-{config.folding_algorithm}.json"), 'r') as f:
            struc_data = json.load(f)
        return struc_data["structure"], struc_data["loop_type"]
    except FileNotFoundError:
        return None, None


if __name__ == '__main__':
    from utils import set_project_path

    FILE_NAME = "codon"
    CHECK_REPRODUCTION = True

    dev_config = OmegaConf.create(
        {"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
         "batch_size": 32, "num_workers": 4})
    set_project_path(dev_config)

    get_train_data_file(FILE_NAME, check_reproduce=CHECK_REPRODUCTION)
