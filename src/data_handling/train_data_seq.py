import os
import json
import pickle
from tqdm import tqdm
import torch
import numpy as np
from omegaconf import OmegaConf

from data_handling.train_val_test_indices import get_train_val_test_indices, get_train_val_test_indices_from_file
from data_handling.data_utils import store_data, check_identical

MAX_SEQ_LENGTH = 9000  # Maximum number of codons in CDS (note: 3' and 5' tails (UTR) are removed)
MAX_DATA = 300_000  # 182_625 seq-tuple pairs in total
FOLDING_ALG = "linearfold"
train_identifiers_path = "/export/share/krausef99dm/data/data_train/codon_train_2.7k_indices.csv"
val_identifiers_path = "/export/share/krausef99dm/data/data_test/codon_val_2.7k_indices.csv"
test_identifiers_path = "/export/share/krausef99dm/data/data_test/codon_test_2.7k_indices.csv"
SEED = 1192  # randomly drawn with np.random.randint(0,2024) on 22.10.2024, 15:00

TOKENS = "01235ACGT().BEHIMSX"


def _get_structure_pred(identifier: str, folding_algorithm="linearfold"):
    try:
        with open(os.path.join(os.environ["PROJECT_PATH"],
                               f"data/sec_struc/{identifier}-{folding_algorithm}.json"), 'r') as f:
            struc_data = json.load(f)
        return struc_data["structure"], struc_data["loop_type"], struc_data["MFE"]
    except FileNotFoundError:
        return None, None, None


def get_train_data_file(file_name: str, check_reproduce=False):
    """Store data for training, validation and testing"""
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    identifiers = []
    sequences = []
    tissue_ids = []
    # mfes = []
    rna_data = []
    targets = []
    targets_bin = []

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

        if len(sequence) > MAX_SEQ_LENGTH:
            continue
        else:
            for tissue_id, target in enumerate(content['targets']):
                if np.isnan(target):
                    continue

                # add 1 to each index to have 0 as encoding for padding
                sequence_ohe = [TOKENS.index(c) + 1 for c in sequence]
                coding_area_ohe = [TOKENS.index(str(int(c))) + 1 for c in coding_area]  # original values: 0,1,2,3,5
                sec_struc_ohe = [TOKENS.index(c) + 1 for c in sec_struc]
                loop_type_ohe = [TOKENS.index(c) + 1 for c in loop_type]

                rna_data.append(torch.tensor([sequence_ohe, coding_area_ohe, sec_struc_ohe, loop_type_ohe]))  # 4 x n

                identifiers.append(identifier)
                sequences.append(sequence)
                tissue_ids.append(tissue_id)
                # mfes.append(mfe)
                targets.append(target)
                targets_bin.append(int(data_targets_bin[tissue_id]))

                if len(rna_data) >= MAX_DATA:
                    break
            if len(rna_data) >= MAX_DATA:
                break

    rna_data = [ts.permute(1, 0) for ts in rna_data]  # change rna_data to n x 4

    # NOTE: can create train-val-test split with either random_state or from codon split files for max comparability
    # indices = get_train_val_test_indices(sequences, random_state=SEED)
    indices = get_train_val_test_indices_from_file(identifiers, train_identifiers_path, val_identifiers_path,
                                                   test_identifiers_path)
    train_indices, val_indices, test_indices = indices

    print("Num seq-tuple pairs TRAIN:", len(train_indices))
    print("Num seq-tuple pairs VAL:", len(val_indices))
    print("Num seq-tuple pairs TEST:", len(test_indices))

    max_seq_len_logging = str(MAX_SEQ_LENGTH / 1000) + "k"

    # assert len(train_indices) > 1000, "Not enough data for training"
    # assert len(val_indices) > 100, "Not enough data for validation"
    # assert len(test_indices) > 100, "Not enough data for testing"

    if check_reproduce:
        check_identical(train_indices, identifiers, tissue_ids,
                        f"data/data_train/{file_name}train_{max_seq_len_logging}")
        check_identical(val_indices, identifiers, tissue_ids,
                        f"data/data_test/{file_name}val_{max_seq_len_logging}")
        check_identical(test_indices, identifiers, tissue_ids,
                        f"data/data_test/{file_name}test_{max_seq_len_logging}")
    else:
        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, train_indices,
                   f"data/data_train/{file_name}train_{max_seq_len_logging}")

        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, val_indices,
                   f"data/data_test/{file_name}val_{max_seq_len_logging}")

        store_data(identifiers, rna_data, tissue_ids, targets, targets_bin, test_indices,
                   f"data/data_test/{file_name}test_{max_seq_len_logging}")

        print("Data successfully created")


if __name__ == '__main__':
    from utils import set_project_path

    CHECK_REPRODUCTION = True
    FILE_NAME = ""

    if FILE_NAME:
        FILE_NAME += "_"

    dev_config = OmegaConf.create(
        {"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
         "batch_size": 32, "num_workers": 4, "train_data_file": "NONONONO", "dev": True, "binary_class": False,
         "frequency_features": False})
    set_project_path(dev_config)

    get_train_data_file(FILE_NAME, check_reproduce=CHECK_REPRODUCTION)
