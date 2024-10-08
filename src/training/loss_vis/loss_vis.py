import datetime
import os.path

import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(subproject, run, ylim=1):
    log_path = os.path.join("/export/share/krausef99dm/runs/", SUBPROJECT, RUN, "weights/losses_fold-0.csv")

    loss_df = pd.read_csv(log_path, index_col=0)
    loss_df.columns = ['epoch', 'train_loss', 'val_loss', 'stored']
    val_loss_subset = loss_df[loss_df['val_loss'].notnull()]

    plt.figure(figsize=(12, 8))
    plt.plot(loss_df['epoch'], loss_df['train_loss'], label='train_loss')
    plt.plot(val_loss_subset['epoch'], val_loss_subset['val_loss'], label='val_loss', marker='o')

    plt.ylim(0, ylim)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(log_path)
    plt.tight_layout()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'training/loss_vis/plots/loss_plot_{SUBPROJECT}_{RUN}_{timestamp}.png')


if __name__ == '__main__':
    SUBPROJECT = 'dev_xlstm'
    RUN = '2_test_150'
    YLIM = 2

    plot_loss(SUBPROJECT, RUN, YLIM)
