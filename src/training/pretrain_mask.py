import torch
from omegaconf import DictConfig


def get_pretrain_mask(data, config: DictConfig):
    # TODO ASSURE THIS WORKS FOR INPUT (B, N, D)
    rna_data = data[0]

    # NAIVE MASKING FOR DEBUGGING
    # also wrong, as padding will also be masked
    # make first 10 elements random and rest 0
    mask = torch.zeros(rna_data[:, :, 0].shape)
    mask[:, :10] = torch.randint(0, 2, (rna_data.shape[0], 10))
    # need a mask of shape (B, N)



    mask = mask.bool()

    return mask


def base_level_masking():
    pass


def subsequence_masking():
    pass


def motif_level_masking():
    pass
