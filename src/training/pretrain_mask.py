import torch
import random
from omegaconf import OmegaConf, DictConfig

from data_handling.train_data_seq import TOKENS
MASK_TOKEN = len(TOKENS) + 1


def get_pretrain_mask(data, config: DictConfig):
    # data is list of [rna_data: list, tissue_ids: tensor, seq_lengths: tensor]
    # with rna_data (list of tensors) = (B, N, D)

    # TODO make sure to handle different sequence lengths
    # masking_strategy = random.choice([naive_masking, base_level_masking, subsequence_masking, motif_level_masking])
    masking_strategy = naive_masking
    return masking_strategy(data, config)


def naive_masking(data, config: DictConfig):
    # Mask 10% of the tokens randomly, only within the sequence length for each sample in the batch
    batch_size, max_len, dim = data[0].shape
    seq_lengths = data[2]  # (batch,)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=data[0].device)

    mask_fraction = 0.05
    for i in range(batch_size):
        length = seq_lengths[i].item()
        num_to_mask = int(length * mask_fraction)
        if num_to_mask > 0:
            positions = torch.randperm(length)[:num_to_mask]
            mask[i, positions] = True

    # Extract original target tokens
    targets = data[0][mask].permute(1, 0)  # shape: (dim, total_masked_positions)

    # Replace masked positions with MASK_TOKEN
    data[0][mask] = MASK_TOKEN

    return data, targets, mask


def base_level_masking(data, config: DictConfig):
    # TODO BERT based
    data, targets, mask = None, None, None

    return data, targets, mask


def subsequence_masking(data, config: DictConfig):
    # TODO ERNIE based
    data, targets, mask = None, None, None

    return data, targets, mask


def motif_level_masking():
    # TODO MOTIF based
    data, targets, mask = None, None, None

    return data, targets, mask


if __name__ == "__main__":
    # For testing
    import copy

    config_dev = OmegaConf.load("config/PTRnet.yml")
    config_dev = OmegaConf.merge(config_dev, {"binary_class": True, "max_seq_length": 1000,
                                              "embedding_max_norm": 2, "gpu_id": 0, "pretrain": True,
                                              "max_seq_length": 9000})

    sequences, seq_lengths = [], []
    for i in range(config_dev.batch_size):
        length = torch.randint(100, config_dev.max_seq_length + 1, (1,)).item()
        # Generate each column with its own integer range
        col0 = torch.randint(low=1, high=6, size=(length,))  # values in [1,5]
        col1 = torch.randint(low=6, high=11, size=(length,))  # values in [6,10]
        col2 = torch.randint(low=11, high=14, size=(length,))  # values in [11,13]
        col3 = torch.randint(low=14, high=20, size=(length,))  # values in [14,19]

        seq = torch.stack([col0, col1, col2, col3], dim=-1)
        sequences.append(seq)
        seq_lengths.append(length)

    sample_batch = [
        torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True),  # rna_data_padded (batch_size x max_seq_length)
        torch.randint(29, (config_dev.batch_size,)),  # tissue_ids (batch_size x 1)
        torch.tensor(seq_lengths, dtype=torch.int64)  # seq_lengths (batch_size x 1)
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
