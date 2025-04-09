import os
import torch
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.knowledge_db import CODON_MAP_DNA, TISSUES


def fit_evaluate_simple_models(train_dataset, val_dataset, binary_class=False):
    """ FOR DEVELOPMENT ONLY
    Computes a cv score for baseline models (RandomForest) using the frequency of codons and tissue one-hot encoding
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
        y.append(dataset.targets_bin.tolist())
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


def store_data(identifiers: list, rna_data: list, target_ids: list, targets: list, targets_bin: list, indices: list,
               path: str):
    identifiers_selected = [identifiers[i] for i in indices]
    rna_data_selected = [rna_data[i] for i in indices]
    target_ids_selected = [target_ids[i] for i in indices]
    targets_selected = [targets[i] for i in indices]
    targets_bin_selected = [targets_bin[i] for i in indices]
    with open(os.path.join(os.environ["PROJECT_PATH"], path + "_data.pkl"), 'wb') as f:
        pickle.dump([rna_data_selected, torch.tensor(target_ids_selected, dtype=torch.int8),
                     torch.tensor(targets_selected), torch.tensor(targets_bin_selected, dtype=torch.int8)], f)
    pd.DataFrame({"identifier": identifiers_selected, "target_id": target_ids_selected, "index": indices}).to_csv(
        os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"), index=False)


def check_identical(indices: list, identifiers: list, target_ids: list, path: str):
    """Check reproducibility of data split"""
    identifiers_selected = [identifiers[i] for i in indices]
    target_ids_selected = [target_ids[i] for i in indices]

    selected_set = set(zip(identifiers_selected, target_ids_selected))
    try:
        persistence = pd.read_csv(os.path.join(os.environ["PROJECT_PATH"], path + "_indices.csv"))
        persistence_set = set(zip(persistence["identifier"].tolist(), persistence["target_id"].tolist()))
        if selected_set != persistence_set:
            raise Exception("REPRODUCTION ISSUE DETECTED")
        else:
            print("Reproducibility check passed!")
    except FileNotFoundError:
        print("Warning: No persistence file found")
