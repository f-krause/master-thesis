# Create train, validation and test data for binary classification task (codon & nucleotide level)

import os
import pickle
import torch
import json
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.knowledge_db import CODON_MAP_DNA, TOKENS
from train_val_test_indices import get_train_val_test_indices
from data_utils import store_data, check_identical

MAX_SEQ_LENGTH_CODON = 8_100  # Maximum number of nucleotides in CDS (note: 3' and 5' tails (UTR) are removed)
MAX_SEQ_LENGTH_NUCLE = 9_000  # Maximum number of nucleotides in CDS (note: 3' and 5' tails (UTR) are removed)
MAX_DATA = 300_000  # 182_625 seq-tuple pairs in total # FIXME
SEED = 1192  # randomly drawn with np.random.randint(0,2024) on 22.10.2024, 15:00
FOLDING_ALG = "linearfold"  # only supported algorithm


def _get_structure_pred(identifier: str, folding_algorithm="linearfold"):
    try:
        with open(os.path.join(os.environ["PROJECT_PATH"],
                               f"data/sec_struc/{identifier}-{folding_algorithm}.json"), 'r') as f:
            struc_data = json.load(f)
        return struc_data["structure"], struc_data["loop_type"], struc_data["MFE"]
    except FileNotFoundError:
        return None, None, None


def get_train_data_file(file_name: str, check_reproduce=False, balanced=False):
    """Store data for training, validation and testing.
    The data is processed from the raw data, split stratified by sequence length and in a way to assure that identical
    mRNA sequences are always in the same set.
    """
    # check if file_name path already exists
    max_seq_len_logging = str(MAX_SEQ_LENGTH_NUCLE / 1000) + "k"
    if os.path.isfile(os.path.join(os.environ["PROJECT_PATH"], f"data/data_train/{file_name}train_{max_seq_len_logging}_data.pkl")) or \
            os.path.isfile(os.path.join(os.environ["PROJECT_PATH"], f"data/data_test/{file_name}val_{max_seq_len_logging}_data.pkl")) or \
            os.path.isfile(os.path.join(os.environ["PROJECT_PATH"], f"data/data_test/{file_name}test_{max_seq_len_logging}_data.pkl")):
        print("WARNING: Data files already exist. Files might be overwritten!")

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
        sec_struc, loop_type, mfe = _get_structure_pred(identifier, FOLDING_ALG)
        if sec_struc is None:
            # logger.warning(f"Skipping {identifier}: no structure prediction found")
            continue

        sequence = content['fasta']
        coding_area = content['bed_annotation']

        # value: 0, low-PTR: 1, high-PTR: 2
        data_targets_bin = (np.where(np.isnan(content["targets"]), np.nan, 0) +
                            np.where(np.isnan(content["targets_bin"]), 0, content["targets_bin"] + 1))

        coding_sequence = [nucleotide for nucleotide, annotation in zip(list(sequence), coding_area) if
                           annotation not in [5, 3]]  # CDS, drop 5' and 3' UTR
        encoded_sequence = [CODON_MAP_DNA.get(''.join(coding_sequence[i:i + 3]), -1) for i in
                            range(0, len(coding_sequence), 3)]

        if len(coding_sequence) > MAX_SEQ_LENGTH_CODON and len(sequence) > MAX_SEQ_LENGTH_NUCLE:
            continue
        else:
            for tissue_id, target in enumerate(content['targets']):
                if np.isnan(target):
                    continue
                if data_targets_bin[tissue_id] < 1:
                    continue

                # add 1 to each index to have 0 as encoding for padding
                sequence_ohe = [TOKENS.index(c) + 1 for c in sequence]
                coding_area_ohe = [TOKENS.index(str(int(c))) + 1 for c in coding_area]  # original values: 0,1,2,3,5
                sec_struc_ohe = [TOKENS.index(c) + 1 for c in sec_struc]
                loop_type_ohe = [TOKENS.index(c) + 1 for c in loop_type]

                nucleotide_rna_data = torch.tensor([sequence_ohe, coding_area_ohe, sec_struc_ohe, loop_type_ohe],
                                                   dtype=torch.int8)  # 4 x n
                nucleotide_rna_data = nucleotide_rna_data.permute(1, 0)  # change rna_data to n x 4
                codon_rna_data = torch.tensor(encoded_sequence)

                identifiers.append(identifier)
                sequences.append("".join(coding_sequence))
                tissue_ids.append(tissue_id)
                targets.append(target)
                targets_bin.append(int(data_targets_bin[tissue_id]))
                rna_data.append({"nucleotide_rna_data": nucleotide_rna_data, "codon_rna_data": codon_rna_data})

                if len(rna_data) >= MAX_DATA:
                    break
            if len(rna_data) >= MAX_DATA:
                break

    if balanced:
        train_indices, val_indices, test_indices = get_train_val_test_indices(sequences, tissue_ids, targets_bin,
                                                                              random_state=SEED)
    else:
        train_indices, val_indices, test_indices = get_train_val_test_indices(sequences, random_state=SEED)

    print("Num seq-tuple pairs TRAIN:", len(train_indices))
    print("Num seq-tuple pairs VAL:", len(val_indices))
    print("Num seq-tuple pairs TEST:", len(test_indices))

    # sanity checks
    assert max([len(s) for s in sequences]) <= MAX_SEQ_LENGTH_NUCLE
    assert max([len(s["codon_rna_data"]) for s in rna_data]) <= MAX_SEQ_LENGTH_CODON

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

    CHECK_REPRODUCTION = True
    BALANCED = True
    FILE_NAME = "binary_class_balanced"

    if FILE_NAME:
        FILE_NAME += "_"

    dev_config = OmegaConf.create(
        {"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline", "pretrain": False,
         "batch_size": 32, "num_workers": 4, "train_data_file": "", "dev": True, "binary_class": False,
         "frequency_features": False})
    set_project_path(dev_config)

    get_train_data_file(FILE_NAME, check_reproduce=CHECK_REPRODUCTION, balanced=BALANCED)
