import os
import torch
import time
import aim  # https://aimstack.io/#demos
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf, DictConfig
from knockknock import discord_sender

from log.logger import setup_logger
from utils import save_checkpoint, mkdir, get_device, get_model_stats, clean_model_weights
from models.get_model import get_model
from data_handling.data_loader import get_train_data_loaders
from training.optimizer import get_optimizer
from training.early_stopper import EarlyStopper
from evaluation.predict import evaluate


def train_fold(config: DictConfig, fold: int = 0):
    logger = setup_logger()

    # Initialize Aim run
    aim_run = aim.Run(experiment=os.environ["SUBPROJECT"].replace("runs/", "", 1),
                      repo="~/master-thesis", log_system_params=True)
    aim_run['model_config'] = OmegaConf.to_container(config)

    # gpu selection
    device = get_device(config, logger)
    # device = torch.device("cpu")  # FIXME for development

    # Create checkpoint directory
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "weights")
    mkdir(checkpoint_path)
    logger.info(f"Checkpoint path: {checkpoint_path}")

    model = get_model(config, device, logger)
    optimizer = get_optimizer(model, config.optimizer)

    if config.binary_class:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()

    early_stopper = EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_delta)
    train_loader, val_loader = get_train_data_loaders(config, fold=fold)

    losses = {}

    logger.info("Starting training")
    start_time = time.time()
    end_time = None
    best_epoch = 1
    y_true_train_best, y_pred_train_best = None, None
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
            aim_run.track(train_neg_auc, name="train_neg_auc", epoch=epoch)

        logger.info(f'Epoch {epoch}, Loss: {train_loss}')
        aim_run.track(train_loss, name='train_loss', epoch=epoch)

        # Save checkpoint
        if (epoch % config.save_freq == 0 and epoch >= config.warmup) or epoch == config.epochs:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(checkpoint_path, f'checkpoint_{epoch}_fold-{fold}.pth.tar'))
            losses[epoch].update({"checkpoint_stored": 1})
            logger.info(f'Checkpoint saved at epoch {epoch}')
            aim_run.track(1, name='checkpoint_stored', epoch=epoch)

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

            logger.info(f'Validation loss: {val_loss}')
            aim_run.track(val_loss, name='val_loss', epoch=epoch)

            if config.binary_class:
                val_neg_auc = -roc_auc_score(y_true_val, y_pred_val)
                losses[epoch].update({"val_neg_auc": val_neg_auc})
                logger.info(f'Validation neg AUC:  {val_neg_auc}')
                aim_run.track(val_neg_auc, name="val_neg_auc", epoch=epoch)
                val_loss = val_neg_auc  # for early stopping in classification setting
                if val_neg_auc < losses[best_epoch].get("val_neg_auc", 0):
                    best_epoch = epoch
                    end_time = time.time()
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                    y_true_train_best, y_pred_train_best = y_true.flatten(), y_pred.flatten()
            else:
                if val_loss < losses[best_epoch].get("val_loss", np.inf):
                    best_epoch = epoch
                    end_time = time.time()
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                    y_true_train_best, y_pred_train_best = y_true.flatten(), y_pred.flatten()

            if early_stopper.early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                aim_run.track(1, name='early_stopping', epoch=epoch)
                break

    if not end_time:
        end_time = time.time()
    aim_run.track(1, name='training_successful')

    # Save losses to a CSV file
    pd.DataFrame(losses).T.to_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))

    # Track times
    training_time = round((end_time - start_time) / 60, 4)
    logger.info(f"Training process completed. Training time for best model: {training_time} mins.")
    aim_run.track(training_time, name='training_time_min')
    aim_run.track(training_time / best_epoch, name='avg_epoch_time')

    if config.model != "dummy" and config.model != "best" and config.model != "mamba2":
        nr_params, nr_flops = get_model_stats(config, model, device, logger)
        aim_run.track(nr_params, name='nr_params')
        aim_run.track(nr_flops, name='nr_flops')

    if config.final_evaluation:
        evaluate(y_true_val_best, y_pred_val_best, "val", best_epoch, config.binary_class,
                 os.environ["SUBPROJECT"], logger, aim_run)
        evaluate(y_true_train_best, y_pred_train_best, "train", best_epoch, config.binary_class,
                 os.environ["SUBPROJECT"], logger, aim_run)

    if config.clean_up_weights:
        clean_model_weights(best_epoch, fold, checkpoint_path, logger)

    logger.info(f"Weights path: {checkpoint_path}")
    aim_run.close()


@discord_sender(webhook_url="https://discord.com/api/webhooks/1308890399942774946/"
                            "3UQa1CD1iNt1JRccZUxPj8ksKJzSuYcAnXMSYa8l9H4gs1DYi-t64qUR8o9-J4A1NFzS")
def train(config: DictConfig):
    # TODO possibility of parallelization across folds!
    for fold in range(config.nr_folds):
        train_fold(config, fold)

    return {"run": os.environ["SUBPROJECT"].replace("runs/", "", 1)}


# TODO include optuna wrapper?
# https://github.com/optuna/optuna/blob/master/README.md#key-features
