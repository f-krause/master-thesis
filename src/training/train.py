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
from utils import save_checkpoint, mkdir, get_device, get_model_stats, log_pred_true_scatter, log_confusion_matrix, \
    log_roc_curve
from models.get_model import get_model
from data_handling.data_loader import get_train_data_loaders
from training.optimizer import get_optimizer
from training.early_stopper import EarlyStopper
from sklearn.metrics import roc_auc_score
from evaluation.predict import predict_and_evaluate


def train_fold(config: DictConfig, fold: int = 0):
    logger = setup_logger()

    # Initialize Aim run
    aim_run = aim.Run(experiment=os.environ["SUBPROJECT"].replace("runs/", "", 1),
                      repo="~/master-thesis", log_system_params=True)
    aim_run['model_config'] = OmegaConf.to_container(config)

    # gpu selection
    device = get_device(config, logger)
    # device = "cpu"  # FIXME for development

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
    last_epoch = 0
    for epoch in range(1, config.epochs + 1):
        last_epoch = epoch
        model.train()
        running_loss = 0.0
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

        train_loss = running_loss / len(train_loader)
        losses[epoch] = {"epoch": epoch, "train_loss": train_loss}

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
        targets, predictions = [], []
        if epoch % config.val_freq == 0 or epoch % config.save_freq == 0 or epoch == config.epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                # TODO add loop for train data
                for data, target, target_bin in val_loader:
                    data = [d.to(device) for d in data]
                    if config.binary_class:
                        target = target_bin
                    target = target.to(device)

                    output = model(data)
                    loss = criterion(output.squeeze().float(), target.float())
                    val_loss += loss.item()

                    predictions.append(output.cpu().numpy())
                    targets.append(target.unsqueeze(1).cpu().numpy())
            val_loss /= len(val_loader)
            losses[epoch].update({"val_loss": val_loss})
            predictions = np.vstack(predictions)
            targets = np.vstack(targets)

            logger.info(f'Validation loss: {val_loss}')
            aim_run.track(val_loss, name='val_loss', epoch=epoch)
            if config.binary_class:
                neg_auc = -roc_auc_score(targets, predictions)
                losses[epoch].update({"val_loss": val_loss, "val_neg_auc": neg_auc})
                logger.info(f'Validation neg AUC:  {neg_auc}')
                aim_run.track(neg_auc, name="val_neg_auc", epoch=epoch)
                val_loss = neg_auc  # for early stopping in classification setting

            if early_stopper.early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                aim_run.track(1, name='early_stopping', epoch=epoch)
                break

    end_time = time.time()
    aim_run.track(1, name='training_successful')

    # Save losses to a CSV file
    pd.DataFrame(losses).T.to_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))
    training_time = round((end_time - start_time) / 60, 4)
    logger.info(f"Training process completed. Training time: {training_time} mins.")
    aim_run.track(training_time, name='training_time_min')
    aim_run.track(training_time / last_epoch, name='avg_epoch_time')

    if config.model != "dummy" and config.model != "best" and config.model != "mamba2":
        nr_params, nr_flops = get_model_stats(config, model, device, logger)
        aim_run.track(nr_params, name='nr_params')
        aim_run.track(nr_flops, name='nr_flops')

    # TODO refactor, evaluation basically already happens during training - no need to load data again and compute AUC!
    if config.final_evaluation:
        logger.info("Starting prediction and evaluation")
        y_true, y_pred, metric = predict_and_evaluate(config, os.environ["SUBPROJECT"], logger)
        img_buffer = log_pred_true_scatter(y_true, y_pred, config.binary_class)
        aim_run.track(aim.Image(img_buffer), name="pred_true_scatter")

        if config.binary_class:
            aim_run.track(metric, name='metric_AUC')
            img_buffer = log_confusion_matrix(y_true, y_pred)
            aim_run.track(aim.Image(img_buffer), name="confusion_matrix")
            img_buffer = log_roc_curve(y_true, y_pred)
            aim_run.track(aim.Image(img_buffer), name="roc_curve")
        else:
            aim_run.track(metric, name='metric_R2')

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
