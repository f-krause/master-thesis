import datetime
import pandas as pd
import matplotlib.pyplot as plt


LOSS_FILE = '/export/share/krausef99dm/runs/dev_baseline/test_run_300/weights/losses_fold-0.csv'


def plot_loss(log_path, ylim=1):
    loss_df = pd.read_csv(log_path)
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
    plt.savefig(f'training/loss_vis/plots/loss_plot_{timestamp}.png')


if __name__ == '__main__':
    plot_loss(LOSS_FILE, 1)
