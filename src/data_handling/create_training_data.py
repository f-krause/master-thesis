import os
import json
import pickle
import numpy as np
from box import Box

tokens = 'ACGT().BEHIMSX'


def get_train_data_file(config: Box):
    with open(os.path.join(os.environ["PROJECT_PATH"], "data/ptr_data.pkl"), 'rb') as f:
        raw_data = pickle.load(f)

    data = []
    targets = []

    counter = 0
    for identifier, content in raw_data.items():
        sec_struc, loop_type = _get_structure_pred(identifier, config)
        if sec_struc is None:
            # logger.warning(f"Skipping {identifier}: no structure prediction found")
            continue

        sequence = content['fasta']
        for target_id, target in enumerate(content['targets']):
            if np.isnan(target):
                continue
            try:
                data.append({
                    'seq': [tokens.index(c) + 1 for c in sequence],  # one hot encoded, 0 is for padding
                    'sec_struc': [tokens.index(c) + 1 for c in sec_struc],
                    'loop_type': [tokens.index(c) + 1 for c in loop_type],
                    'target_id': target_id,
                })
                targets.append(target)
            except ValueError:
                print(sequence)
            counter += 1
            if counter >= 10:
                break

    with open(os.path.join(os.environ["PROJECT_PATH"], "data/dev_train_data_small.pkl"), 'wb') as f:
        pickle.dump([data, targets], f)

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

    config = Box({"project_path": None, "log_file_path": None, "subproject": "dev", "model": "baseline",
                  "batch_size": 32, "num_workers": 4, "folding_algorithm": "viennarna"})
    set_project_path(config)
    set_log_file(config)
    get_train_data_file(config)
    print("Data successfully created")
