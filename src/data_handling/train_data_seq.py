# TODO implement as train_data_codons.py but for sequences
import os
import json
import pickle
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

MAX_SEQ_LENGTH = 8100  # Maximum sequence length (nr of bases of whole mRNA)
MAX_DATA = 1000
FOLDING_ALG = "viennarna"
TOKENS_BASES = 'ACGT'
TOKENS_STRUC = '().'
TOKENS_LOOP = 'BEHIMSX'


def get_train_data_file(config: DictConfig, return_dict=False):
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    rna_data = []
    targets = []
    target_ids = []

    for identifier, content in raw_data.items():
        sec_struc, loop_type = _get_structure_pred(identifier, config)
        if sec_struc is None:
            # logger.warning(f"Skipping {identifier}: no structure prediction found")
            continue

        sequence = content['fasta']
        if len(sequence) > MAX_SEQ_LENGTH:
            continue

        for target_id, target in enumerate(content['targets']):
            if np.isnan(target):
                continue

            numeric_data = {
                'seq': [TOKENS_BASES.index(c) + 1 for c in sequence],  # label encoded, 0 is reserved for padding
                'sec_struc': [TOKENS_STRUC.index(c) + 1 for c in sec_struc],
                'loop_type': [TOKENS_LOOP.index(c) + 1 for c in loop_type],
                'target_id': target_id,
            }

            if return_dict:
                rna_data.append(numeric_data)
                targets.append(target)
            else:
                # create matrix 3 x n: Seq, SecStruc, LoopType
                tensor_data = torch.tensor([numeric_data["seq"], numeric_data["sec_struc"], numeric_data["loop_type"]])
                tensor_data = tensor_data.permute(1, 0)
                rna_data.append(tensor_data)
                target_ids.append(target_id)
                targets.append(target)

            if len(rna_data) >= MAX_DATA:
                break
        if len(rna_data) >= MAX_DATA:
            break

    # full data
    # TODO add train test split as for codons
    # with open(os.path.join(os.environ["PROJECT_PATH"], "data/data_train/dev_train_data_small.pkl"), 'wb') as f:
    #     pickle.dump([rna_data, torch.tensor(target_ids), torch.tensor(targets)], f)


def _get_structure_pred(identifier: str, config: DictConfig):
    try:
        with open(os.path.join(os.environ["PROJECT_PATH"],
                               f"data/sec_struc/{identifier}-{config.folding_algorithm}.json"), 'r') as f:
            struc_data = json.load(f)
        return struc_data["structure"], struc_data["loop_type"]
    except FileNotFoundError:
        return None, None


if __name__ == '__main__':
    from utils import set_project_path, set_log_file

    dev_config = OmegaConf.create({"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
                  "batch_size": 32, "num_workers": 4, "folding_algorithm": FOLDING_ALG})
    set_project_path(dev_config)
    set_log_file(dev_config)
    get_train_data_file(dev_config)
    print("Data successfully created")
