# adapted from https://github.com/CatIIIIIIII/RNAErnie/blob/main/rna_pretrainer.py (last accessed 10.12.2024)
import os
from ahocorapy.keywordtree import KeywordTree

from data_handling.train_data_seq import TOKENS


def seq2kmer(seq, k_mer, max_length=None):
    kmer_text = ""
    i = 0
    upper = len(seq) - k_mer + 1
    if max_length:
        upper = min(upper, max_length)
    while i < upper:
        kmer_text += (seq[i: i + k_mer] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


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


if __name__ == "__main__":
    motif_tree_dict = get_motif_tree_dict()
    print(motif_tree_dict)