import os
from ahocorapy.keywordtree import KeywordTree  # TODO: install ahocorapy


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


def load_motif(motif_dir, motif_name, tokenizer):
    """load motifs from file

    Args:
        motif_dir (str): motif data root directory
        motif_name (str): motif file name
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer

    Returns:
        dict: {file_name: [int]}
    """
    res = {}
    for name in motif_name.split(","):
        motif_path = os.path.join(motif_dir, name + ".txt")
        with open(motif_path, 'r') as f:
            motifs = f.readlines()

        motif_tokens = []
        for m in motifs:
            kmer_text = seq2kmer(seq=m, k_mer=1)
            input_ids = tokenizer(kmer_text, return_token_type_ids=False)[
                "input_ids"]
            input_ids = input_ids[1:-1]
            motif_tokens.append(input_ids)
        res[name] = motif_tokens

    return res

def get_motif_dict():
    motif_dict = load_motif(motif_dir=args.motif_dir,
                            motif_name=args.motif_files,
                            tokenizer=tokenizer)

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