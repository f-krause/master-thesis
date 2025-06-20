# ideas from https://github.com/CatIIIIIIII/RNAErnie/blob/main/rna_pretrainer.py (last accessed 10.12.2024)
import torch
import random
import pickle
from omegaconf import OmegaConf, DictConfig

from data_handling.train_data_seq import TOKENS
from pretraining.pretrain_utils import hash_sequence, get_motif_tree_dict
from pretraining.store_identified_motifs import precompute_motif_matches
from utils.knowledge_db import CODON_MAP_DNA

MASK_TOKEN = len(TOKENS) + 1


def naive_masking(data, config: DictConfig):
    # Mask 5% of the tokens randomly (just as an example)
    rna_data, tissue_ids, seq_lengths, freq = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = min(length, config.nr_masked_tokens)
        if num_to_mask > 0:
            positions = torch.randperm(length, device=device)[:num_to_mask]
            mask[i, positions] = True

    # Extract original target tokens before modifying rna_data
    # shape after masking: (total_masked_positions, dim)
    targets = rna_data[mask].permute(1, 0).long()  # (dim, total_masked_positions)

    # Replace masked positions with MASK_TOKEN (for all features)
    rna_data[mask] = MASK_TOKEN

    return [rna_data, tissue_ids, seq_lengths, freq], targets, mask


def apply_masking_strategy(rna_data, mask, device):
    # Apply the 80/10/10 replacements
    num_masked = mask.sum().item()
    if num_masked > 0:
        # Decide replacement strategy for each masked position
        rand_replacements = torch.rand(num_masked, device=device)

        # Column-specific token ranges (adjust if needed)
        col_ranges = [
            (6, 10),   # tokens in [6,9] - seq ohe
            (1, 6),    # tokens in [1,5] - coding area
            (10, 13),  # tokens in [10,12] - loop type pred
            (13, 20),  # tokens in [13,19] - sec structure pred
        ]

        # Create random tokens aligned with column ranges
        random_cols = []
        for low, high in col_ranges:
            # draw random tokens within [low, high-1]
            col_random = torch.randint(low=low, high=high, size=(num_masked,), device=device)
            random_cols.append(col_random)
        random_tokens = torch.stack(random_cols, dim=-1).int()  # (num_masked, dim)

        # Masks for each strategy
        mask80 = rand_replacements < 0.8   # 80% -> [MASK]
        mask10 = rand_replacements >= 0.9  # 10% random

        rna_data_copy = rna_data.clone()
        mutated_rna_mask = rna_data_copy[mask]
        mutated_rna_mask[mask80] = MASK_TOKEN  # Apply 80%: [MASK] token
        mutated_rna_mask[mask10] = random_tokens[mask10]  # 10%: random token
        # remaining 10% of masked: original (do nothing)

        rna_data[mask] = mutated_rna_mask

    return rna_data


def base_level_masking(data, config: DictConfig, motif_cache=None, motif_tree_dict=None):
    # BERT-style masking:
    # 1. ~15% of tokens are selected for masking
    # 2. Of these selected:
    #    80% -> [MASK]
    #    10% -> original token
    #    10% -> random token from the vocab (per-column aligned)
    rna_data, tissue_ids, seq_lengths, freq = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = min(length, config.nr_masked_tokens)
        if num_to_mask > 0:
            positions = torch.randperm(length, device=device)[:num_to_mask]
            mask[i, positions] = True

    # Extract the original tokens that will be predicted
    # targets: (dim, total_masked_positions)
    targets = rna_data[mask].permute(1, 0).clone().long()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    # Return mutated data, targets, and mask
    return [rna_data, tissue_ids, seq_lengths, freq], targets, mask


def subsequence_masking(data, config: DictConfig, motif_cache=None, motif_tree_dict=None):
    # ERNIE-like approach:
    # Instead of single tokens, we mask contiguous subsequences.
    # We'll:
    #  - Select ~15% of tokens to predict.
    #  - Split these tokens into a few contiguous subsequences.
    #  - Mask each subsequence with the same 80/10/10 rule as above.
    rna_data, tissue_ids, seq_lengths, freq = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    # We'll choose subsequence lengths randomly (e.g., lengths 2-5) until we reach num_to_mask
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = min(length, config.nr_masked_tokens)
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
    targets = rna_data[mask].permute(1, 0).clone().long()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    return [rna_data, tissue_ids, seq_lengths, freq], targets, mask


def motif_level_masking(data, config, motif_cache, motif_tree_dict):
    # data = [rna_data: (B, N, D), tissue_ids: (B,), seq_lengths: (B,)]
    rna_data, tissue_ids, seq_lengths, freq = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    # Extract tokens and move to CPU once
    token_data_cpu = rna_data[..., 0].cpu()

    # Pre-initialize the mask on GPU
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        length = seq_lengths[i].item()
        if length == 0:
            continue

        # Identify sequence by a known ID or hash. Here we assume `sequence_ids[i]`
        # was passed in and matches what we used in precomputation.
        sequence = token_data_cpu[i, :length]

        # Retrieve precomputed candidates
        seq_hash = hash_sequence(sequence)
        seq_hash_reversed = hash_sequence(sequence.flip(0))
        if seq_hash in motif_cache["DataBases"].keys():
            ngram_candidates_db = motif_cache["DataBases"].get(seq_hash, [])
            ngram_candidates_stat = motif_cache["Statistics"].get(seq_hash, [])
        elif seq_hash_reversed in motif_cache["DataBases"].keys():
            ngram_candidates_db = motif_cache["DataBases"].get(seq_hash_reversed, [])
            ngram_candidates_stat = motif_cache["Statistics"].get(seq_hash_reversed, [])
        else:
            print("WARNING: No motifs found in cache, computing on-the-fly")
            temp_generation = precompute_motif_matches([sequence], motif_tree_dict)  # this is expensive
            ngram_candidates_db = temp_generation["DataBases"].get(seq_hash, [])
            ngram_candidates_stat = temp_generation["Statistics"].get(seq_hash, [])
            motif_cache["DataBases"][seq_hash] = ngram_candidates_db  # Update cache
            motif_cache["Statistics"][seq_hash] = ngram_candidates_stat

        random.shuffle(ngram_candidates_db)
        random.shuffle(ngram_candidates_stat)

        ngram_candidates = ngram_candidates_db + ngram_candidates_stat
        if not ngram_candidates:
            print("WARNING: No motifs found for sequence")
            continue

        num_to_predict = min(length, config.nr_masked_tokens)

        # Select motifs without overlap until we reach num_to_predict
        covered_indexes = set()
        selected_indexes = []
        tokens_chosen = 0
        for ngram in ngram_candidates:
            if tokens_chosen >= num_to_predict:
                break
            if tokens_chosen + len(ngram) > num_to_predict:
                needed = num_to_predict - tokens_chosen
                available = [idx for idx in ngram if idx < length and idx not in covered_indexes]
                if len(available) >= needed:
                    selected = random.sample(available, needed)
                    covered_indexes.update(selected)
                    selected_indexes.extend(selected)
                    tokens_chosen += needed
                continue
            if any(idx in covered_indexes for idx in ngram):
                continue
            covered_indexes.update(ngram)
            selected_indexes.extend(ngram)
            tokens_chosen += len(ngram)

        if selected_indexes:
            # Ensure we're not out of range (accounting for shorter sequences)
            selected_indexes = [idx for idx in selected_indexes if idx < length]
            if selected_indexes:
                mask[i, selected_indexes] = True

        # print(mask.sum())

    # If too few positions got masked, fallback to base-level masking
    if mask.sum() < 10:
        print("WARNING: Too few masked entries, falling back to base-level masking instead")
        return base_level_masking(data, config)

    targets = rna_data[mask].permute(1, 0).clone().long()

    # Apply the 80/10/10 replacements
    rna_data = apply_masking_strategy(rna_data, mask, device)

    return [rna_data, tissue_ids, seq_lengths, freq], targets, mask


def get_pretrain_mask_data(epoch, data, config: DictConfig, motif_cache, motif_tree_dict):
    # data: [rna_data: (B,N,D), tissue_ids: (B,), seq_lengths: (B,)]
    # motif masking 10 times slower than others, hence in total around 3 times slower as without
    if epoch < config.efficient_masking_epochs:
        masking_strategy = random.choice([base_level_masking, subsequence_masking])
    else:
        masking_strategy = random.choice([base_level_masking, subsequence_masking, motif_level_masking])

    # masking_strategy = motif_level_masking  # for dev
    return masking_strategy(data, config, motif_cache, motif_tree_dict)


if __name__ == "__main__":
    # For testing
    import copy

    config_dev = OmegaConf.create({
        "batch_size": 8,
        "efficient_masking_epochs": 0,
        "max_seq_length": 1000,
        "binary_class": True,
        "embedding_max_norm": 2,
        "gpu_id": 0,
        "pretrain": True,
        "nr_masked_tokens": 10,
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
        torch.tensor(seq_lengths, dtype=torch.int64),  # seq_lengths (B)
        torch.randn(config_dev.batch_size, len(CODON_MAP_DNA)),  # frequency_features (B x 64)
    ]
    sample_batch_copy = copy.deepcopy(sample_batch)

    # Single masking strategies testing
    motif_cache, motif_tree_dict = None, None
    masking_base_level = base_level_masking(sample_batch, config_dev, motif_cache, motif_tree_dict)
    # masking_subseq = subsequence_masking(sample_batch, config_dev, motif_cache, motif_tree_dict)

    with open("/export/share/krausef99dm/data/data_train/motif_matches_cache.pkl", 'rb') as f:
        motif_cache = pickle.load(f)
    motif_tree_dict = get_motif_tree_dict()

    masking_motif = motif_level_masking(sample_batch, config_dev, motif_cache, motif_tree_dict)

    # full pipeline testing
    mutated_data, targets, mask = get_pretrain_mask_data(100, sample_batch, config_dev, motif_cache,
                                                         motif_tree_dict)

    print(mutated_data[0].shape)
    print(mask.shape)
    print(targets.shape)
    assert mutated_data[0].shape == sample_batch_copy[0].shape
    assert mask.shape == sample_batch_copy[0][:, :, 0].shape
    assert targets.shape[1] == mask.sum()
    print("Test passed")
