import os
import pickle
import torch
import pandas as pd


def main(data_path, seq_file_name, cod_file_name):
    seq_indices_path = os.path.join(data_path, seq_file_name + "_indices.csv")
    cod_indices_path = os.path.join(data_path, cod_file_name + "_indices.csv")

    indices_seq = pd.read_csv(seq_indices_path).identifier
    indices_seq_set = set(indices_seq.tolist())

    indices_cod = pd.read_csv(cod_indices_path).identifier
    indices_cod_set = set(indices_cod.tolist())

    indices_to_remove = (indices_seq_set | indices_cod_set) - (indices_seq_set & indices_cod_set)
    nr_indices_to_remove = len(indices_to_remove)

    print("Union of unique identifiers:       ", len(indices_seq_set | indices_cod_set))
    print("Intersection of unique identifiers:", len(indices_seq_set & indices_cod_set))
    print("len seq test:", len(indices_seq_set))
    print("len cod test:", len(indices_cod_set))
    print("indices to remove:", indices_to_remove)
    print("len indices to remove:", nr_indices_to_remove)

    if nr_indices_to_remove > 0:
        print("Removing sequences and storing new files")
        seq_path = os.path.join(data_path, seq_file_name + "_data.pkl")
        cod_path = os.path.join(data_path, cod_file_name + "_data.pkl")

        with open(seq_path, "rb") as f:
            seq_data = pickle.load(f)

        with open(cod_path, "rb") as f:
            cod_data = pickle.load(f)

        seq_bool_keep = ~torch.tensor(indices_seq.isin(indices_to_remove))
        cod_bool_keep = ~torch.tensor(indices_cod.isin(indices_to_remove))

        rna_seq_data = [[data for i, data in enumerate(seq_data[0]) if seq_bool_keep[i]]]
        seq_data = rna_seq_data + [data[seq_bool_keep] for data in seq_data[1:]]

        rna_cod_data = [[data for i, data in enumerate(cod_data[0]) if cod_bool_keep[i]]]
        cod_data = rna_cod_data + [data[cod_bool_keep] for data in cod_data[1:]]

        indices_seq_df = pd.read_csv(seq_indices_path)[seq_bool_keep.tolist()]
        indices_cod_df = pd.read_csv(cod_indices_path)[cod_bool_keep.tolist()]

        # check if lengths now align
        assert len(seq_data[0]) == len(cod_data[0])
        assert len(seq_data[1]) == len(cod_data[1])
        assert len(seq_data[2]) == len(cod_data[2])
        assert len(seq_data[3]) == len(cod_data[3])
        assert len(indices_seq_df) == len(indices_cod_df)
        assert len(seq_data[0]) == len(seq_data[1]) == len(seq_data[2]) == len(seq_data[3]) == len(indices_seq_df)
        assert len(cod_data[0]) == len(cod_data[1]) == len(cod_data[2]) == len(cod_data[3]) == len(indices_cod_df)

        # overwrite files
        if input(
                f"Do you want to remove {nr_indices_to_remove} indices and overwrite files to make codon and nucleotide "
                f"level sets identical? [y/n]").lower() == "y":
            with open(seq_path, "wb") as f:
                pickle.dump(seq_data, f)

            with open(cod_path, "wb") as f:
                pickle.dump(cod_data, f)

            # overwrite csv file
            indices_seq_df.to_csv(seq_indices_path, index=False)
            indices_cod_df.to_csv(cod_indices_path, index=False)

            print("Files overwritten.")
    else:
        print("Codon and nucleotide level sets are identical")


if __name__ == "__main__":
    data_path = "/export/share/krausef99dm/data/data_test/"
    seq_file_name = "test_9.0k"
    cod_file_name = "codon_test_8.1k"

    main(data_path, seq_file_name, cod_file_name)
