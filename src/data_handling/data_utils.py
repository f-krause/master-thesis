import torch
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from knowledge_db import CODON_MAP_DNA, TISSUES


def fit_evaluate_simple_models(train_dataset, val_dataset):
    """ FOR DEVELOPMENT ONLY """
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

    x, y = [], []
    for dataset in [train_dataset, val_dataset]:
        for tissue_id, rna in zip(dataset.tissue_ids, dataset.rna_data):
            counts = torch.bincount(rna, minlength=len(CODON_MAP_DNA) + 1)[1:len(CODON_MAP_DNA) + 1]
            freq = counts.float() / counts.sum()
            tissue_one_hot = [1 if i == tissue_id else 0 for i in range(len(TISSUES))]
            x.append(freq.tolist() + tissue_one_hot)
        y.append(dataset.targets_bin.tolist())
    y = y[0] + y[1]

    for clf in [clf1, clf2]:
        cv_scores = cross_validate(clf, x, y, cv=5, scoring=['roc_auc'], return_train_score=True)
        print(type(clf).__name__)
        print("Mean Test ROC:", cv_scores['test_roc_auc'].mean())
        print(cv_scores)