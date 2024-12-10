# ideas from https://github.com/CatIIIIIIII/RNAErnie/blob/main/rna_pretrainer.py (last accessed 10.12.2024)
import torch
import random
from omegaconf import OmegaConf, DictConfig

from data_handling.train_data_seq import TOKENS
from pretraining.pretrain_utils import get_motif_tree_dict

MASK_TOKEN = len(TOKENS) + 1
motif_tree_dict = get_motif_tree_dict()  # ahocorapy KeywordTree with encoded motifs as values


def naive_masking(data, config: DictConfig):
    # Mask 5% of the tokens randomly (just as an example)
    rna_data, tissue_ids, seq_lengths = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    mask_fraction = 0.05
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = int(length * mask_fraction)
        if num_to_mask > 0:
            positions = torch.randperm(length, device=device)[:num_to_mask]
            mask[i, positions] = True

    # Extract original target tokens before modifying rna_data
    # shape after masking: (total_masked_positions, dim)
    targets = rna_data[mask].permute(1, 0)  # (dim, total_masked_positions)

    # Replace masked positions with MASK_TOKEN (for all features)
    rna_data[mask] = MASK_TOKEN

    return [rna_data, tissue_ids, seq_lengths], targets, mask


def apply_masking_strategy(rna_data, mask, device):
    # Apply the 80/10/10 replacements
    num_masked = mask.sum().item()
    if num_masked > 0:
        # Decide replacement strategy for each masked position
        rand_replacements = torch.rand(num_masked, device=device)

        # Column-specific token ranges (adjust if needed)
        col_ranges = [
            (6, 10),  # tokens in [6,9] - seq ohe
            (1, 6),   # tokens in [1,5] - coding area
            (10, 13), # tokens in [10,12] - loop type pred
            (13, 20), # tokens in [13,19] - sec structure pred
        ]

        # Create random tokens aligned with column ranges
        random_cols = []
        for low, high in col_ranges:
            # draw random tokens within [low, high-1]
            col_random = torch.randint(low=low, high=high, size=(num_masked,), device=device)
            random_cols.append(col_random)
        random_tokens = torch.stack(random_cols, dim=-1)  # (num_masked, dim)

        # Masks for each strategy
        mask80 = rand_replacements < 0.8   # 80% -> [MASK]
        mask10 = rand_replacements >= 0.9  # 10% random

        mutated_rna_mask = rna_data[mask]
        mutated_rna_mask[mask80] = MASK_TOKEN  # Apply 80%: [MASK] token
        mutated_rna_mask[mask10] = random_tokens[mask10]  # 10%: random token
        # remaining 10% of masked: original (do nothing)

        rna_data[mask] = mutated_rna_mask

    return rna_data


def base_level_masking(data, config: DictConfig):
    # BERT-style masking:
    # 1. ~15% of tokens are selected for masking
    # 2. Of these selected:
    #    80% -> [MASK]
    #    10% -> original token
    #    10% -> random token from the vocab (per-column aligned)

    rna_data, tissue_ids, seq_lengths = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    masked_lm_prob = 0.15  # TODO add to config
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = int(length * masked_lm_prob)
        if num_to_mask > 0:
            positions = torch.randperm(length, device=device)[:num_to_mask]
            mask[i, positions] = True

    # Extract the original tokens that will be predicted
    # targets: (dim, total_masked_positions)
    targets = rna_data[mask].permute(1, 0).clone()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    # Return mutated data, targets, and mask
    return [rna_data, tissue_ids, seq_lengths], targets, mask


def subsequence_masking(data, config: DictConfig):
    # ERNIE-like approach:
    # Instead of single tokens, we mask contiguous subsequences.
    # We'll:
    #  - Select ~15% of tokens to predict.
    #  - Split these tokens into a few contiguous subsequences.
    #  - Mask each subsequence with the same 80/10/10 rule as above.

    rna_data, tissue_ids, seq_lengths = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    masked_lm_prob = 0.15  # TODO add to config
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    # We'll choose subsequence lengths randomly (e.g., lengths 2-5) until we reach num_to_mask
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = int(length * masked_lm_prob)
        if num_to_mask > 0:
            tokens_to_mask = 0
            while tokens_to_mask < num_to_mask:
                # Random subsequence length
                subseq_len = random.randint(3, 9)
                start = random.randint(0, length - subseq_len)
                end = start + subseq_len
                # Avoid exceeding num_to_mask
                if tokens_to_mask + subseq_len > num_to_mask:
                    # Just mask the needed remainder
                    subseq_len = num_to_mask - tokens_to_mask
                    end = start + subseq_len

                # Set mask True for this subsequence
                mask[i, start:end] = True
                tokens_to_mask += subseq_len

    # Targets are original tokens at masked positions
    targets = rna_data[mask].permute(1, 0).clone()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    return [rna_data, tissue_ids, seq_lengths], targets, mask


def motif_level_masking(data, config: DictConfig):
    # data = [rna_data: (B, N, D), tissue_ids: (B,), seq_lengths: (B,)]
    # 1. Search motifs using the motif_trees["DataBases"] and motif_trees["Statistics"].
    # 2. Gather the found motifs as ngram candidates.
    # 3. Shuffle them and select as many as needed until we reach num_to_predict.
    # 4. Mask them with 80/10/10 distribution, choosing random tokens column-wise.

    rna_data, tissue_ids, seq_lengths = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    # Hyperparameters (can be from config)
    masked_lm_prob = getattr(config, 'masked_lm_prob', 0.15)
    max_predictions_per_seq = getattr(config, 'max_predictions_per_seq', int(max_len * masked_lm_prob))

    # Prepare mask
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    # Motif searching: We'll consider the first column of rna_data as "tokens"
    # If another column is needed, adjust here.
    for i in range(batch_size):
        length = seq_lengths[i].item()
        if length == 0:
            continue

        # Extract tokens for motif search (as a Python list)
        # We're assuming motif trees expect a list of ints
        tokens_list = rna_data[i, :length, 0].tolist()

        # Compute num_to_predict
        num_to_predict = min(max_predictions_per_seq, max(1, int(round(length * masked_lm_prob))))

        # Find motif occurrences in "DataBases"
        motif_tree_db = motif_tree_dict["DataBases"]
        db_results = motif_tree_db.search_all(tokens_list)
        # db_results is a list of tuples like: (matched_string, start_index)
        # We'll store the indices covered by this motif
        ngram_indexes = []
        for result in db_results:
            motif_len = len(result[0])
            start_idx = result[1]
            ngram_index = list(range(start_idx, start_idx + motif_len))
            ngram_indexes.append([ngram_index])

        random.shuffle(ngram_indexes)  # Shuffle the candidates

        # If we haven't reached num_to_predict, also look at "Statistics"
        motif_tree_stats = motif_tree_dict["Statistics"]
        stats_results = motif_tree_stats.search_all(tokens_list)
        ngram_indexes_extra = []
        for result in stats_results:
            motif_len = len(result[0])
            start_idx = result[1]
            ngram_index = list(range(start_idx, start_idx + motif_len))
            ngram_indexes_extra.append([ngram_index])

        random.shuffle(ngram_indexes_extra)  # Shuffle the candidates
        ngram_indexes.extend(ngram_indexes_extra)  # Statistics motifs are added if not enough from Databases

        # Select motifs until we reach num_to_predict
        covered_indexes = set()
        selected_indexes = []
        for cand_index_set in ngram_indexes:
            if len(selected_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            index_set = cand_index_set[0]
            # Check if adding this motif exceeds num_to_predict
            if len(selected_indexes) + len(index_set) > num_to_predict:
                continue
            # Check overlap
            if any(idx in covered_indexes for idx in index_set):
                continue
            # Add these indexes
            for idx in index_set:
                covered_indexes.add(idx)
            selected_indexes.extend(index_set)

        # Mark them in the mask
        # Only mask within actual length
        selected_indexes = [idx for idx in selected_indexes if idx < length]  # not sure if needed
        if len(selected_indexes) > 0:
            mask[i, selected_indexes] = True

    # Extract targets (original tokens at masked positions)
    targets = rna_data[mask].permute(1, 0).clone()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    return [rna_data, tissue_ids, seq_lengths], targets, mask


def get_pretrain_mask_data(data, config: DictConfig):
    # data: [rna_data: (B,N,D), tissue_ids: (B,), seq_lengths: (B,)]
    masking_strategy = random.choice([base_level_masking, subsequence_masking, motif_level_masking])
    # masking_strategy = motif_level_masking  # for dev
    return masking_strategy(data, config)


if __name__ == "__main__":
    # For testing
    import copy

    config_dev = OmegaConf.create({
        "batch_size": 8,
        "max_seq_length": 1000,
        "binary_class": True,
        "embedding_max_norm": 2,
        "gpu_id": 0,
        "pretrain": True,
    })

    sequences, seq_lengths = [], []
    for i in range(config_dev.batch_size):
        length = torch.randint(100, config_dev.max_seq_length + 1, (1,)).item()
        # Generate each column with its own integer range
        col0 = torch.randint(low=6, high=10, size=(length,))  # values in [6,9] - seq ohe
        col1 = torch.randint(low=1, high=6, size=(length,))  # values in [1,5] - coding area
        col2 = torch.randint(low=10, high=13, size=(length,))  # values in [10,12] - loop type pred
        col3 = torch.randint(low=13, high=20, size=(length,))  # values in [13,19] - sec structure pred

        seq = torch.stack([col0, col1, col2, col3], dim=-1)
        sequences.append(seq)
        seq_lengths.append(length)

    sample_batch = [
        torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),  # rna_data_padded (B x N x D)
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (B)
        torch.tensor(seq_lengths, dtype=torch.int64)  # seq_lengths (B)
    ]
    sample_batch_copy = copy.deepcopy(sample_batch)
    mutated_data, targets, mask = get_pretrain_mask_data(sample_batch, config_dev)
    print(mutated_data[0].shape)
    print(mask.shape)
    print(targets.shape)
    assert mutated_data[0].shape == sample_batch_copy[0].shape
    assert mask.shape == sample_batch_copy[0][:, :, 0].shape
    assert targets.shape[1] == mask.sum()
    print("Test passed")
