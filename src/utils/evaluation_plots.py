import io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay


def log_pred_true_scatter(y_true, y_pred, tissue_ids, limits, binary_class=False):
    selected_tissues = np.array(range(limits[0], limits[1]))  # Convert to NumPy array for efficient filtering

    # Filter y_true, y_pred, and tissue_ids
    mask = np.isin(tissue_ids, selected_tissues)
    if sum(mask) == 0:
        # handle regression case without tissue_ids
        mask = np.ones_like(mask)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    tissue_ids = np.array(tissue_ids)[mask]

    # Generate plot
    plt.figure(figsize=(10, 10))

    # Generate y_true (adding a small uniform noise) and y_pred
    if binary_class:
        y_true = y_true + np.random.uniform(0, 0.8, len(y_true))

    # Get unique tissue ids and sort them
    unique_tissues = sorted(np.unique(tissue_ids))

    # Define a list of markers to cycle through
    markers = ['o', 's', '^', 'D']  # , 'v', 'P', '*', 'X', 'h', 'H', '8', 'p', 'd']
    cmap = plt.get_cmap('nipy_spectral', len(unique_tissues))

    for i, tissue in enumerate(unique_tissues):
        mask = tissue_ids == tissue
        plt.scatter(
            y_true[mask],
            y_pred[mask],
            label=f'{tissue}',
            alpha=0.5,
            s=40,
            c=[cmap(i)],
            marker=markers[i % len(markers)]
        )

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('y_true vs y_pred by Selected Tissue IDs')
    plt.legend(title='Tissue ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    if binary_class:
        # min_val = min(np.min(y_true), np.min(y_pred))
        min_val = np.min(y_true)
        # max_val = max(np.max(y_true), np.max(y_pred))
        max_val = np.max(y_true)
        plt.plot([min_val, max_val], [0.5, 0.5], color='grey', linestyle='--')

    plt.tight_layout()

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)


def log_confusion_matrix(y_true, y_pred):
    y_pred = [1 if target > 0.5 else 0 for target in y_pred]  # make target binary, tau = 0.5

    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)


def log_roc_curve(y_true, y_pred):
    plt.figure(figsize=(10, 10))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='estimator')
    disp.plot()
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)