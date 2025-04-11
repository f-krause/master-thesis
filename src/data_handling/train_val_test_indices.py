import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from collections import defaultdict

from utils.knowledge_db import TISSUES


# Define helper function to balance unique sequences at the tissue level.
def _balance_unique_sequences(split_seqs, tissue_ids, targets_bin, seq_to_indices, seed):
    """
    For the given set of unique sequences, determine for each tissue the sequences that would be kept
    in a balanced manner. A unique sequence is retained only if, for every tissue where it appears,
    it falls in the balanced (i.e. sampled) subset.
    """
    # Build per-tissue mapping: for each tissue t, record lists of unique sequences by target.
    tissue_to_seqs = {t: {0: [], 1: []} for t in range(len(TISSUES))}
    for seq in split_seqs:
        # Get all indices (i.e. all tissue assignments) for this sequence.
        indices = seq_to_indices[seq]
        seen_tissues = set()
        for idx in indices:
            t = tissue_ids[idx]
            if t in seen_tissues:
                continue
            # We assume all occurrences in one tissue have the same target.
            tissue_to_seqs[t][targets_bin[idx]].append(seq)
            seen_tissues.add(t)

    # For each tissue, randomly sample from the majority class if needed.
    rng = np.random.default_rng(seed)
    tissue_selected = {}
    for t in range(len(TISSUES)):
        pos = np.array(tissue_to_seqs[t][1])
        neg = np.array(tissue_to_seqs[t][0])
        # Determine the number to keep per class (if both classes are present).
        if len(pos) and len(neg):
            n_keep = min(len(pos), len(neg))
            # If one class is larger, sample to have only n_keep elements.
            if len(pos) > n_keep:
                pos_selected = set(rng.choice(pos, n_keep, replace=False).tolist())
            else:
                pos_selected = set(pos.tolist())
            if len(neg) > n_keep:
                neg_selected = set(rng.choice(neg, n_keep, replace=False).tolist())
            else:
                neg_selected = set(neg.tolist())
            tissue_selected[t] = pos_selected.union(neg_selected)
        else:
            # If one class is missing, we keep all unique sequences for that tissue.
            tissue_selected[t] = set(pos.tolist() + neg.tolist())

    # Now, keep only those unique sequences that, for every tissue in which they appear,
    # are part of the selected set.
    balanced_seqs = []
    for seq in split_seqs:
        indices = seq_to_indices[seq]
        seq_tissues = set(tissue_ids[idx] for idx in indices)
        if all(seq in tissue_selected[t] for t in seq_tissues):
            balanced_seqs.append(seq)
    return balanced_seqs


def get_train_val_test_indices(mrna_sequences, tissue_ids=None, targets_bin=None,
                               val_frac=0.15, test_frac=0.15, num_bins=10, random_state=None):
    """
    Splits mRNA sequences into train, validation, and test sets without overlapping sequences.
    Stratification is based on sequence lengths.
    Also aims to balance the total number of sequence-target pairs across splits.
    If tissue_ids and targets_bin are provided, performs additional balancing per tissue:
    for each tissue, a random subset of the mRNA sequences from the majority class is dropped such that
    the number of positive and negative samples is identical.

    Parameters:
      - mrna_sequences (list): List of coding mRNA sequences (a given sequence may repeat if it has multiple targets).
      - tissue_ids (list): List of tissue IDs (0-28) for each sequence.
      - targets_bin (list): List of binary targets for each sequence (low/high PTR).
      - val_frac (float): Fraction of data for validation set.
      - test_frac (float): Fraction of data for test set.
      - num_bins (int): Number of bins for stratification based on sequence length.
      - random_state (int): Seed for reproducibility.

    Returns:
      - train_indices (list): Sorted list of indices for training set.
      - val_indices (list): Sorted list of indices for validation set.
      - test_indices (list): Sorted list of indices for test set.
    """
    train_frac = 1 - val_frac - test_frac

    # Map each unique sequence to all its indices.
    seq_to_indices = defaultdict(list)
    for idx, seq in enumerate(mrna_sequences):
        seq_to_indices[seq].append(idx)

    unique_seqs = list(seq_to_indices.keys())
    counts_per_seq = {seq: len(indices) for seq, indices in seq_to_indices.items()}
    # Sanity check: each mRNA sequence appears no more than the total number of tissues.
    # (Assumes global variable TISSUES holds list of tissues; adjust as needed.)
    # assert max(counts_per_seq.values()) <= len(TISSUES)

    seq_lengths = [len(seq) for seq in unique_seqs]
    # Get histogram bins based on sequence lengths
    bins = np.histogram(seq_lengths, bins=num_bins)[1]
    bins = bins[:-1]  # use left edges
    seq_length_bins = np.digitize(seq_lengths, bins)

    # Lists that will hold the unique sequences for each split.
    train_seqs = []
    val_seqs = []
    test_seqs = []

    # Initialize a temporary random seed counter; we change seed for each bin to make shuffling independent.
    temp_random_state = random_state if random_state is not None else 0

    # For each length bin, partition sequences according to cumulative counts.
    for bin_num in np.unique(seq_length_bins):
        # Get indices (into unique_seqs) of sequences in this bin.
        bin_indices = [i for i, bin_idx in enumerate(seq_length_bins) if bin_idx == bin_num]
        bin_seqs = [unique_seqs[i] for i in bin_indices]
        bin_counts = [counts_per_seq[seq] for seq in bin_seqs]
        bin_total_count = sum(bin_counts)

        # Determine desired number of sequence-target pairs per set for this bin.
        bin_desired_train_count = bin_total_count * train_frac
        bin_desired_val_count = bin_total_count * val_frac if test_frac > 0 else bin_total_count

        # Shuffle sequences within the bin
        temp_random_state += 1
        rng = np.random.default_rng(seed=temp_random_state)
        shuffle_indices = rng.permutation(len(bin_seqs))
        bin_seqs_shuffled = [bin_seqs[i] for i in shuffle_indices]
        bin_counts_shuffled = [bin_counts[i] for i in shuffle_indices]

        cumulative_counts = np.cumsum(bin_counts_shuffled)
        train_threshold = bin_desired_train_count
        val_threshold = bin_desired_train_count + bin_desired_val_count

        bin_train_seqs = []
        bin_val_seqs = []
        bin_test_seqs = []

        for seq, cum_count in zip(bin_seqs_shuffled, cumulative_counts):
            if cum_count <= train_threshold:
                bin_train_seqs.append(seq)
            elif cum_count <= val_threshold:
                bin_val_seqs.append(seq)
            else:
                bin_test_seqs.append(seq)

        train_seqs.extend(bin_train_seqs)
        val_seqs.extend(bin_val_seqs)
        test_seqs.extend(bin_test_seqs)

    # Convert the unique sequences (for each split) into full indices.
    # If tissue_ids and targets_bin are provided, apply tissue balancing.
    if tissue_ids is not None and targets_bin is not None:
        targets_bin = [idx - 1 for idx in targets_bin]  # change scale to 0 and 1

        train_seqs_balanced = _balance_unique_sequences(train_seqs, tissue_ids, targets_bin, seq_to_indices,
                                                       random_state)
        val_seqs_balanced = _balance_unique_sequences(val_seqs, tissue_ids, targets_bin, seq_to_indices, random_state)
        # test_seqs_balanced = _balance_unique_sequences(test_seqs, tissue_ids, targets_bin, seq_to_indices, random_state)

        train_indices = [idx for seq in train_seqs_balanced for idx in seq_to_indices[seq]]
        val_indices = [idx for seq in val_seqs_balanced for idx in seq_to_indices[seq]]
        test_indices = [idx for seq in test_seqs for idx in seq_to_indices[seq]]  # FIXME balancing unnecessary
    else:
        train_indices = [idx for seq in train_seqs for idx in seq_to_indices[seq]]
        val_indices = [idx for seq in val_seqs for idx in seq_to_indices[seq]]
        test_indices = [idx for seq in test_seqs for idx in seq_to_indices[seq]]

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    return train_indices, val_indices, test_indices


def get_train_val_test_indices_from_file(identifiers, train_identifiers_path, val_identifiers_path,
                                         test_identifiers_path):
    # load identifiers
    train_ids = set(pd.read_csv(train_identifiers_path).identifier.tolist())
    val_ids = set(pd.read_csv(val_identifiers_path).identifier.tolist())
    test_ids = set(pd.read_csv(test_identifiers_path).identifier.tolist())

    # get indices
    train_indices = [i for i, identifier in enumerate(identifiers) if identifier in train_ids]
    val_indices = [i for i, identifier in enumerate(identifiers) if identifier in val_ids]
    test_indices = [i for i, identifier in enumerate(identifiers) if identifier in test_ids]

    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    from utils.utils import set_project_path

    dev_config = OmegaConf.create({"project_path": None, "subproject": "dev"})
    set_project_path(dev_config)

    np.random.seed(42)
    seq_lengths_dummy = np.random.randint(0, 8000, 100000)
    train, val, test = get_train_val_test_indices(seq_lengths_dummy, random_state=40)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    # show distribution of sequence lengths in each set
    print(np.histogram(seq_lengths_dummy, bins=10, density=True))
    print(np.histogram(seq_lengths_dummy[train], bins=10, density=True))
    print(np.histogram(seq_lengths_dummy[val], bins=10, density=True))
    print(np.histogram(seq_lengths_dummy[test], bins=10, density=True))

    train2, val2, test2 = get_train_val_test_indices(seq_lengths_dummy, random_state=40)
    assert np.array_equal(train, train2)
    assert np.array_equal(val, val2)
    assert np.array_equal(test, test2)
    print("Seed test passed")
