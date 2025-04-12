import pickle
from tqdm import tqdm

from pretraining.pretrain_utils import hash_sequence


def precompute_motif_matches(sequences, motif_tree, store: bool = False, cache_path=None):
    """
    Precompute motif matches for all sequences in the dataset.
    :param sequences: An iterable that yields (seq_id, tokens_list).
                    seq_id should uniquely identify the sequence.
    :param motif_tree: Pre-initialized database tree for motifs.
    :param cache_path: Where to store the precomputed matches.
    """
    cache = {"DataBases": {}, "Statistics": {}}

    for seq in tqdm(sequences, desc="Precomputing motif matches"):
        # Run motif searches once per sequence
        seq_ls = seq.tolist()
        db_results = motif_tree["DataBases"].search_all(seq_ls)
        stats_results = motif_tree["Statistics"].search_all(seq_ls)

        # Convert results to a list of n-gram indices
        ngram_candidates_db = []
        ngram_candidates_stat = []
        for res in db_results:
            motif_len = len(res[0])
            start_idx = res[1]
            ngram_candidates_db.append(list(range(start_idx, start_idx + motif_len)))
        for res in stats_results:
            motif_len = len(res[0])
            start_idx = res[1]
            ngram_candidates_stat.append(list(range(start_idx, start_idx + motif_len)))

        seq_hash = hash_sequence(seq)
        cache["DataBases"][seq_hash] = ngram_candidates_db
        if len(ngram_candidates_db) < 10:
            print("also adding stats motifs")
            cache["Statistics"][seq_hash] = ngram_candidates_stat
        else:
            cache["Statistics"][seq_hash] = []

    # Save to disk so it can be loaded during training
    if store:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    return cache


if __name__ == '__main__':
    import torch
    from omegaconf import OmegaConf
    from utils.utils import set_project_path
    from pretraining.pretrain_utils import get_motif_tree_dict
    from data_handling.data_loader import RNADataset

    STORE = True
    STORE_PATH = "/export/share/krausef99dm/data/data_train/motif_matches_cache.pkl"

    device = torch.device("cuda:0")
    config_dev = OmegaConf.load("config/PTRnet.yml")
    config_dev_nucl = OmegaConf.load("config/general_nucleotide.yml")
    config_dev = OmegaConf.merge(config_dev, config_dev_nucl)
    config_dev = OmegaConf.merge(config_dev, {"pretrain": False})
    OmegaConf.update(config_dev, "binary_class", False)

    set_project_path(config_dev)

    # Load the dataset
    train_dataset = RNADataset(config_dev, 0)
    val_dataset = RNADataset(config_dev, 0, val=True)

    train_sequences = [seq.permute(1, 0)[0] for seq in train_dataset.rna_data]
    val_sequences = [seq.permute(1, 0)[0] for seq in val_dataset.rna_data]
    sequences = train_sequences + val_sequences

    # make unique
    sequences = list({tuple(tensor.tolist()): tensor for tensor in sequences}.values())

    # Load the motif trees
    motif_tree_dict = get_motif_tree_dict()

    # Precompute the motif matches
    # TODO remove dev and run for full data!
    precompute_motif_matches(sequences, motif_tree_dict, STORE, STORE_PATH)
