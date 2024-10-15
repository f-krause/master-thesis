import os
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


def get_train_val_test_indices(sequence_lengths, train_size=0.7, random_state=42, num_bins=10):
    # TODO add persistency here, store indices in a file
    # indices_path = os.path.join(os.environ["PROJECT_PATH"], "data/.pkl")

    length_bins = np.digitize(sequence_lengths, np.histogram_bin_edges(sequence_lengths, bins=num_bins))

    train_indices, temp_indices = train_test_split(np.arange(len(sequence_lengths)), test_size=1-train_size,
                                                   stratify=length_bins, random_state=random_state)

    temp_bins = length_bins[temp_indices]
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=temp_bins,
                                                 random_state=random_state)

    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    from utils import set_project_path, set_log_file
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

