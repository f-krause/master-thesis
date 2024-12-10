import torch
import random
from omegaconf import OmegaConf, DictConfig

from data_handling.train_data_seq import TOKENS

MASK_TOKEN = len(TOKENS) + 1
motif_tree = None


def get_pretrain_mask(data, config: DictConfig):
    # data: [rna_data: (B,N,D), tissue_ids: (B,), seq_lengths: (B,)]
    # Choose a masking strategy
    # Possible strategies:
    #  - naive_masking
    #  - base_level_masking (BERT-style)
    #  - subsequence_masking (ERNIE-style)
    #  - motif_level_masking
    # For demonstration, we'll just pick one.
    # You could randomize if desired:
    # masking_strategy = random.choice([base_level_masking, subsequence_masking, motif_level_masking])
    masking_strategy = base_level_masking
    return masking_strategy(data, config)


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

    masked_lm_prob = 0.15  # BERT default
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
    num_masked = mask.sum().item()
    if num_masked > 0:
        # Decide replacement strategy for each masked position
        # shape: (num_masked,)
        rand_replacements = torch.rand(num_masked, device=device)

        # Column-specific token ranges (adjust if needed)
        col_ranges = [
            (6, 10),   # tokens in [6,9] - seq ohe
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

    masked_lm_prob = 0.15
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    # We'll choose subsequence lengths randomly (e.g., lengths 2-5) until we reach num_to_mask
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = int(length * masked_lm_prob)
        if num_to_mask > 0:
            tokens_to_mask = 0
            while tokens_to_mask < num_to_mask:
                # Random subsequence length
                subseq_len = random.randint(2, 5)
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

    # Same 80/10/10 distribution as BERT
    num_masked = mask.sum().item()
    if num_masked > 0:
        rand_replacements = torch.rand(num_masked, device=device)
        random_tokens = torch.randint(
            low=1,
            high=len(TOKENS) + 1,
            size=(num_masked, dim),
            device=device
        )

        masked_data_view = rna_data[mask]

        mask80 = rand_replacements < 0.8
        mask10_2 = rand_replacements >= 0.9
        # 80% -> MASK_TOKEN
        masked_data_view[mask80] = MASK_TOKEN
        # 10% -> keep original (do nothing)
        masked_data_view[mask10_2] = random_tokens[mask10_2]

    return [rna_data, tissue_ids, seq_lengths], targets, mask


def motif_level_masking(data, config: DictConfig):
    # Placeholder for motif-based masking.
    # Here, you'd need a list of motifs and their positions for each sequence.
    # Similar to subsequence masking, but we rely on known motifs.
    # Steps:
    #  - Identify motif positions (list of index ranges).
    #  - Select some of them to mask according to masked_lm_prob and max_predictions_per_seq
    #  - Apply 80/10/10 distribution.

    rna_data, tissue_ids, seq_lengths = data
    batch_size, max_len, dim = rna_data.shape
    device = rna_data.device

    masked_lm_prob = 0.15
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    # Suppose we have motif_positions as input (not provided here).
    # motif_positions[i] could be a list of tuples (start, end) for motifs in sequence i.
    # For now, we just do nothing:
    targets = torch.empty(dim, 0, device=device)

    return [rna_data, tissue_ids, seq_lengths], targets, mask


if __name__ == "__main__":
    # For testing
    import copy

    config_dev = OmegaConf.create({
        "batch_size": 4,
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
    mutated_data, targets, mask = get_pretrain_mask(sample_batch, config_dev)
    print(mutated_data[0].shape)
    print(mask.shape)
    print(targets.shape)
    assert mutated_data[0].shape == sample_batch_copy[0].shape
    assert mask.shape == sample_batch_copy[0][:, :, 0].shape
    assert targets.shape[1] == mask.sum()
    print("Test passed")
