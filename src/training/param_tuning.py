import torch
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from omegaconf import DictConfig

from utils import get_device, set_seed
from models.get_model import get_model
from data_handling.data_loader import get_train_data_loaders
from training.optimizer import get_optimizer
from training.early_stopper import EarlyStopper


def train_tune_fold(config: DictConfig, train_loader, val_loader, trial):
    device = get_device(config)
    model = get_model(config, device)
    optimizer = get_optimizer(model, config.optimizer)
    criterion = torch.nn.BCELoss() if config.binary_class else torch.nn.MSELoss()
    early_stopper = EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_delta)

    losses = {}

    best_epoch = 1
    y_true_val_best, y_pred_val_best = None, None
    for epoch in range(1, config.epochs + 1):
        model.train()

        running_loss = 0.0
        y_true, y_pred = [], []
        for batch_idx, (data, target, target_bin) in enumerate(tqdm(train_loader)):
            data = [d.to(device) for d in data]
            if config.binary_class:
                target = target_bin
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze().float(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y_true.append(target.unsqueeze(1).cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        losses[epoch] = {"epoch": epoch, "train_loss": train_loss}
        y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)

        if config.binary_class:
            train_neg_auc = -roc_auc_score(y_true, y_pred)
            losses[epoch].update({"train_neg_auc": train_neg_auc})

        # Validation
        if epoch % config.val_freq == 0 or epoch % config.save_freq == 0 or epoch == config.epochs:
            model.eval()

            val_loss = 0.0
            y_true_val, y_pred_val = [], []
            with torch.no_grad():
                for data, target, target_bin in val_loader:
                    data = [d.to(device) for d in data]
                    if config.binary_class:
                        target = target_bin
                    target = target.to(device)

                    output = model(data)
                    loss = criterion(output.squeeze().float(), target.float())
                    val_loss += loss.item()

                    y_true_val.append(target.unsqueeze(1).cpu().numpy())
                    y_pred_val.append(output.cpu().numpy())

            val_loss /= len(val_loader)
            losses[epoch].update({"val_loss": val_loss})
            y_true_val, y_pred_val = np.vstack(y_true_val), np.vstack(y_pred_val)

            if config.binary_class:
                val_neg_auc = -roc_auc_score(y_true_val, y_pred_val)
                losses[epoch].update({"val_neg_auc": val_neg_auc})
                val_loss = val_neg_auc  # for early stopping in classification setting
                if val_neg_auc < losses[best_epoch].get("val_neg_auc", 0):
                    best_epoch = epoch
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
            else:
                if val_loss < losses[best_epoch].get("val_loss", np.inf):
                    best_epoch = epoch
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()

            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stopper.early_stop(val_loss):
                # TODO needed?
                break

    if config.binary_class:
        return roc_auc_score(y_true_val_best, y_pred_val_best)
    else:
        return root_mean_squared_error(y_true_val_best, y_pred_val_best)


def create_objective(config):
    def objective(trial):
        set_trial_parameters(trial, config)
        set_seed(config.seed)
        train_loader, val_loader = get_train_data_loaders(config, 0)
        try:
            score = train_tune_fold(config, train_loader, val_loader, trial)
        except optuna.exceptions.TrialPruned:
            return None
        return score
    return objective


def set_trial_parameters(trial, config):
    # TODO automize based on yml files
    lr_range = (1e-5, 1e-2)
    # weight_decay_range = (1e-6, 1e-1)
    batch_size_options = [32, 64, 128]
    dropout_range = (0.0, 0.5)
    rnn_hidden_size_options = [64, 128, 256, 512]
    num_layers_range = (1, 4)

    # Suggest hyperparameters
    config.optimizer.lr = trial.suggest_float('lr', *lr_range, log=True)
    # config.optimizer.weight_decay = trial.suggest_float('weight_decay', *weight_decay_range, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', batch_size_options)
    config.dropout = trial.suggest_float('dropout', *dropout_range)
    config.rnn_hidden_size = trial.suggest_categorical('rnn_hidden_size', rnn_hidden_size_options)
    config.num_layers = trial.suggest_int('num_layers', *num_layers_range)
