import os
import json
import pickle
import torch
import numpy as np
from box import Box

from knowledge_db import CODON_MAP_DNA  # FIXME check why data has DNA and not RNA codons (T instead of U)!


MAX_SEQ_LENGTH = 3000  # Maximum number of codons in CDS
MAX_DATA = 1000


def get_train_data_file(config: Box, return_dict=False):
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    rna_data = []
    targets = []
    target_ids = []

    for identifier, content in raw_data.items():
        sequence = content['fasta']

        bed_annotation = content['bed_annotation']
        sequence = list(sequence)
        coding_sequence = [nucleotide for nucleotide, annotation in zip(sequence, bed_annotation) if
                           annotation not in [5, 3]]  # CDS, drop 5' and 3' UTR
        encoded_sequence = [CODON_MAP_DNA.get(''.join(coding_sequence[i:i + 3]), -1) for i in range(0, len(coding_sequence), 3)]

        if len(encoded_sequence) > MAX_SEQ_LENGTH:
            continue

        for target_id, target in enumerate(content['targets']):
            if np.isnan(target):
                continue

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

    with open(os.path.join(os.environ["PROJECT_PATH"], "data/train_data/dev_train_data_1000_max_length.pkl"), 'wb') as f:
        pickle.dump([rna_data, torch.tensor(target_ids), torch.tensor(targets)], f)

    # TODO: Implement test data creation


def _get_structure_pred(identifier: str, config: Box):
    try:
        with open(os.path.join(os.environ["PROJECT_PATH"],
                               f"data/sec_struc/{identifier}-{config.folding_algorithm}.json"), 'r') as f:
            struc_data = json.load(f)
        return struc_data["structure"], struc_data["loop_type"]
    except FileNotFoundError:
        return None, None


if __name__ == '__main__':
    from utils import set_project_path, set_log_file

    dev_config = Box({"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
                      "batch_size": 32, "num_workers": 4})
    set_project_path(dev_config)
    get_train_data_file(dev_config)
    print("Data successfully created")
