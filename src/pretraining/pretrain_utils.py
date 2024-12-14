# adapted from https://github.com/CatIIIIIIII/RNAErnie/blob/main/rna_pretrainer.py (last accessed 10.12.2024)
import os
import hashlib
from ahocorapy.keywordtree import KeywordTree

from data_handling.train_data_seq import TOKENS


def load_motif(motif_name):
    res = {}
    for name in motif_name.split(","):
        motif_path = os.path.join("pretraining/motif_db", name + ".txt")
        with open(motif_path, 'r') as f:
            motifs = f.readlines()
        motifs = [m.replace("\n", "") for m in motifs]
        motifs = [m.replace("U", "T") for m in motifs]

        motif_tokens = []
        for m in motifs:
            input_ids = [TOKENS.index(c) + 1 for c in m]  # tokenize as sequence OHE in train data
            input_ids = input_ids[1:-1]
            motif_tokens.append(input_ids)
        res[name] = motif_tokens

    return res


def get_motif_tree_dict():
    motif_dict = load_motif(motif_name="ATtRACT,SpliceAid,Statistics")

    motif_tree_dict = {}
    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k != "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["DataBases"] = motif_tree

    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k == "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["Statistics"] = motif_tree

    return motif_tree_dict


def hash_sequence(sequence):
    if sequence.device.type != 'cpu':
        sequence = sequence.cpu()
    nonzero_positions = (sequence != 0).nonzero(as_tuple=True)[0]
    first_nonzero_idx = nonzero_positions[0].item()
    last_nonzero_idx = nonzero_positions[-1].item()
    sequence = sequence[first_nonzero_idx:last_nonzero_idx + 1]
    seq_bytes = sequence.numpy().tobytes()
    return hashlib.sha256(seq_bytes).hexdigest()


if __name__ == "__main__":
    # motif_tree_dict = get_motif_tree_dict()
    # print(motif_tree_dict)

    import torch

    sequence = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0], dtype=torch.int8)
    print(hash_sequence(sequence))
