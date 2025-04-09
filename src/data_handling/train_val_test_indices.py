import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from collections import defaultdict

from utils.knowledge_db import TISSUES


def get_train_val_test_indices(mrna_sequences, val_frac=0.15, test_frac=0.15, num_bins=10, random_state=None):
    """
        Splits mRNA sequences into train, validation, and test sets without overlapping sequences.
        Stratification is based on sequence lengths.
        Also aims to balance the total number of sequence-target pairs across splits.

        Parameters:
        - mrna_seq (list): List of coding mRNA sequences (sequences may repeat if they have multiple targets).
        - train_frac (float): Fraction of data for training set.
        - val_frac (float): Fraction of data for validation set.
        - test_frac (float): Fraction of data for test set.

        Returns:
        - train_indices (list): Indices for training set.
        - val_indices (list): Indices for validation set.
        - test_indices (list): Indices for test set.
    """
    train_frac = 1 - val_frac - test_frac

    # Map each unique sequence to its indices and counts
    seq_to_indices = defaultdict(list)
    for idx, seq in enumerate(mrna_sequences):
        seq_to_indices[seq].append(idx)

    unique_seqs = list(seq_to_indices.keys())
    counts_per_seq = {seq: len(indices) for seq, indices in seq_to_indices.items()}
    assert max([counts for counts in counts_per_seq.values()]) <= len(TISSUES)  # sanity check for target count

    seq_lengths = [len(seq) for seq in unique_seqs]

    # Stratify sequences based on length with histogram function for balanced bins
    bins = np.histogram(seq_lengths, bins=num_bins)[1]
    bins = bins[:-1]
    seq_length_bins = np.digitize(seq_lengths, bins)

    train_seqs = []
    val_seqs = []
    test_seqs = []
    temp_random_state = random_state

    # For each bin, assign sequences to sets
    for bin_num in np.unique(seq_length_bins):
        # Get sequences in this bin
        bin_indices = [i for i, bin_idx in enumerate(seq_length_bins) if bin_idx == bin_num]
        bin_seqs = [unique_seqs[i] for i in bin_indices]
        bin_counts = [counts_per_seq[seq] for seq in bin_seqs]

        # Total counts in this bin
        bin_total_count = sum(bin_counts)

        # Desired counts for each set in this bin
        bin_desired_train_count = bin_total_count * train_frac
        if test_frac > 0:
            bin_desired_val_count = bin_total_count * val_frac
        else:
            # If no test set, assign all remaining sequences to validation set
            bin_desired_val_count = bin_total_count

        # Shuffle sequences in the bin
        temp_random_state += 1
        rng = np.random.default_rng(seed=temp_random_state)
        shuffle_indices = rng.permutation(len(bin_seqs))
        bin_seqs_shuffled = [bin_seqs[i] for i in shuffle_indices]
        bin_counts_shuffled = [bin_counts[i] for i in shuffle_indices]

        # Calculate cumulative counts
        cumulative_counts = np.cumsum(bin_counts_shuffled)

        # Determine thresholds
        train_threshold = bin_desired_train_count
        val_threshold = bin_desired_train_count + bin_desired_val_count

        # Assign sequences based on cumulative counts
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

        # Append sequences from this bin to the overall lists
        train_seqs.extend(bin_train_seqs)
        val_seqs.extend(bin_val_seqs)
        test_seqs.extend(bin_test_seqs)

    # Now, collect indices
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
