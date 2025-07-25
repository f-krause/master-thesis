import os
import torch
import pickle
import time
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, accuracy_score,
    precision_score, recall_score, f1_score
)

from utils.knowledge_db import CODON_MAP_DNA, TISSUES, TOKENS


def get_one_hot_tensors(config, rna_data, tissue_ids):
    correction = torch.tensor([6, 1, 10, 13])  # make numerical encodings start from 0
    num_classes = [4, 5, 3, 7]
    if config.align_aug:
        correction = torch.tensor([5, 0, 9, 12])  # make numerical encodings start from 1 (for 5'UTR 0 padding)
        num_classes = [5, 6, 4, 8]  # 0 class as well

    one_hot_tensors = []
    for seq, tissue_id in zip(rna_data, tissue_ids):
        seq = seq - correction
        seq = seq.permute(1, 0)

        new_tensor = []
        for i in range(len(num_classes)):
            one_hot = F.one_hot(seq[i].long(), num_classes=num_classes[i])  # B x L x D x num_classes
            new_tensor.append(one_hot)
        # legacy: also OHE the tissue ID and append
        # new_tensor.append(F.one_hot(tissue_id, num_classes=len(TISSUES)))
        one_hot_tensors.append(torch.hstack(new_tensor))
    return one_hot_tensors


def get_rna_data_kmers(config, rna_data, k=3):
    """ Get k-mers from RNA data with encoding mapping from word2vec model """
    with open("/export/share/krausef99dm/data/w2v_model_data.pkl", 'rb') as f:
        model_data = pickle.load(f)

    rna_seq_only = [rna_data[i][:, 0] for i in range(len(rna_data))]
    int2token = {i + 1: token for i, token in enumerate(TOKENS)}

    rna_data_kmers = []
    for seq in rna_seq_only:
        seq = seq.tolist()
        seq = [int2token.get(i) for i in seq]
        seq_kmers = ["".join(seq[i:i + k]) for i in range(len(seq) - k + 1)]  # FIXME with seq alignment
        if config.align_aug:
            seq_kmers_numeric = [model_data["kmer_to_index"][kmer] + 1 for kmer in seq_kmers]
        else:
            seq_kmers_numeric = [model_data["kmer_to_index"][kmer] for kmer in seq_kmers]
        rna_data_kmers.append(torch.tensor(seq_kmers_numeric))

    return rna_data_kmers


def cv_simple_models(train_dataset, val_dataset, binary_class=False):
    """ FOR DEVELOPMENT ONLY
    Computes a 3-fold cross validation score for baseline models (RandomForest) using the frequency of codons and
    tissue one-hot encoding
    """
    if binary_class:
        clf1 = RandomForestClassifier(random_state=42)
    else:
        clf1 = RandomForestRegressor(random_state=42)

    x, y = [], []
    for dataset in [train_dataset, val_dataset]:
        for tissue_id, rna in zip(dataset.tissue_ids, dataset.rna_data):
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA) + 1)[1:len(CODON_MAP_DNA) + 1]
            freq = counts.float() / counts.sum()
            tissue_one_hot = [1 if i == tissue_id else 0 for i in range(len(TISSUES))]
            x.append(freq.tolist() + tissue_one_hot)
        y.append(dataset.targets_bin.tolist())  # FIXME currently only binary targets supported?
    y = y[0] + y[1]

    for clf in [clf1]:
        if binary_class:
            cv_scores = cross_validate(clf, x, y, cv=3, scoring=['roc_auc'], return_train_score=True)
            print(type(clf).__name__)
            print("Mean Test ROC:", cv_scores['test_roc_auc'].mean(), cv_scores['test_roc_auc'].std())
        else:
            cv_scores = cross_validate(clf, x, y, cv=3, scoring=['neg_root_mean_squared_error'],
                                       return_train_score=True)
            print(type(clf).__name__)
            print("Mean Test RMSE:", cv_scores['test_neg_root_mean_squared_error'].mean())
        print(cv_scores)


def train_validate_simple_model(train_dataset, val_dataset, binary_class=False):
    """ FOR DEVELOPMENT ONLY
    Train on train set, validate on the validation set
    """
    if binary_class:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    def extract_features(dataset):
        x, y = [], []
        for tissue_id, rna, target in zip(dataset.tissue_ids, dataset.rna_data, dataset.targets_bin if binary_class else dataset.targets):
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA) + 1)[1:len(CODON_MAP_DNA) + 1]
            freq = counts.float() / counts.sum()
            tissue_one_hot = [1 if i == tissue_id else 0 for i in range(len(TISSUES))]
            x.append(freq.tolist() + tissue_one_hot)
            y.append(target)
        return x, y

    x_train, y_train = extract_features(train_dataset)
    x_val, y_val = extract_features(val_dataset)

    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    print(type(model).__name__)
    print("Training Time (s):", training_time)

    if binary_class:
        y_train_pred = model.predict(x_train)
        y_val_proba = model.predict_proba(x_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Train Precision:", precision_score(y_train, y_train_pred))
        print("Train Recall:", recall_score(y_train, y_train_pred))
        print("Train F1:", f1_score(y_train, y_train_pred))

        print("Validation ROC AUC:", roc_auc_score(y_val, y_val_proba))
        print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Validation Precision:", precision_score(y_val, y_val_pred))
        print("Validation Recall:", recall_score(y_val, y_val_pred))
        print("Validation F1:", f1_score(y_val, y_val_pred))
    else:
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)

        print("Train RMSE:", mean_squared_error(y_train, y_train_pred, squared=False))
        print("Validation RMSE:", mean_squared_error(y_val, y_val_pred, squared=False))

    return model


def store_data(identifiers: list, rna_data: list, tissue_ids: list, targets: list, targets_bin: list, indices: list,
               path: str):
    identifiers_selected = [identifiers[i] for i in indices]
    rna_data_selected = [rna_data[i] for i in indices]
    tissue_ids_selected = [tissue_ids[i] for i in indices]
    targets_selected = [targets[i] for i in indices]
    targets_bin_selected = [targets_bin[i] for i in indices]
    with open(os.path.join(os.environ["PROJECT_PATH"], path + "_data.pkl"), 'wb') as f:
        pickle.dump([rna_data_selected, torch.tensor(tissue_ids_selected, dtype=torch.int8),
                     torch.tensor(targets_selected), torch.tensor(targets_bin_selected, dtype=torch.int8)], f)
    pd.DataFrame({"identifier": identifiers_selected, "target_id": tissue_ids_selected, "index": indices}).to_csv(
        os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"), index=False)


def check_identical(indices: list, identifiers: list, tissue_ids: list, path: str):
    """Check reproducibility of data split"""
    identifiers_selected = [identifiers[i] for i in indices]
    tissue_ids_selected = [tissue_ids[i] for i in indices]

    selected_set = set(zip(identifiers_selected, tissue_ids_selected))
    try:
        persistence = pd.read_csv(os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"))
        persistence_set = set(zip(persistence["identifier"].tolist(), persistence["target_id"].tolist()))
        if selected_set != persistence_set:
            raise Exception("REPRODUCTION ISSUE DETECTED:", path)
        else:
            print("Reproducibility check passed!")
    except FileNotFoundError:
        print("Warning: No persistence file found")
