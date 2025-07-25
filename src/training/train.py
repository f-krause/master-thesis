import os
import torch
import time
import pickle
import aim  # https://aimstack.io/#demos
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from omegaconf import OmegaConf, DictConfig
from knockknock import discord_sender
from scipy.special import expit

from log.logger import setup_logger
from utils.utils import *
from models.get_model import get_model
from data_handling.data_loader import get_train_data_loaders
from training.optimizer import get_optimizer
from training.early_stopper import EarlyStopper
from pretraining.pretrain_mask import get_pretrain_mask_data
from pretraining.pretrain_utils import get_motif_tree_dict
from evaluation.evaluate import evaluate

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# from training.lr_scheduler import GradualWarmupScheduler, TimmCosineLRScheduler


def train_fold(config: DictConfig, logger, fold: int = 0):
    if config.save_freq % config.val_freq != 0:
        raise ValueError(f"save_freq ({config.save_freq}) should be a multiple of val_freq ({config.val_freq})")

    # Initialize loggers
    experiment_name = os.environ["SUBPROJECT"].replace("runs/", "", 1)
    if config.nr_folds > 1:
        experiment_name += f"_fold-{fold}"
    aim_run = aim.Run(experiment=experiment_name, repo="/export/share/krausef99dm", log_system_params=True)
    aim_run['model_config'] = OmegaConf.to_container(config)

    # gpu selection
    device = get_device(config, logger)
    # device = torch.device("cpu")  # FIXME for development

    # Create checkpoint directory
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "weights")
    mkdir(checkpoint_path)
    logger.info(f"Checkpoint path: {checkpoint_path}")

    # DATA
    train_loader, val_loader = get_train_data_loaders(config, fold=fold)

    scale_data_path = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"])
    if config.scale_targets and fold == 0:  # new flag in config/general_*.yml
        mu, sigma = fit_target_scaler(train_loader.dataset.targets.float())
        logger.info(f"Saving scaler data to {scale_data_path}: mu={mu}, sigma={sigma}")
        torch.save({'mu': mu, 'sigma': sigma}, os.path.join(scale_data_path, 'scaler_data.pt'))

    if config.scale_targets:
        scaler_cfg = torch.load(os.path.join(scale_data_path, 'scaler_data.pt'), weights_only=False)

    # MODEL
    model = get_model(config, device, logger)

    # PRETRAINING
    if config.pretrain_path and not config.pretrain:
        checkpoint = torch.load(config.pretrain_path)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing_keys:
            print("Missing keys:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)

    if config.pretrain:
        logger.info("Loading motif cache")
        if not config.align_aug:
            with open("/export/share/krausef99dm/data/data_train/motif_matches_cache.pkl", 'rb') as f:
                motif_cache = pickle.load(f)
        else:
            # TODO could also build motif cache for AUG aligned sequences
            logger.warning("Motif cache currently not built for aligned sequences, expect slightly slower training.")
            motif_cache = {"DataBases": {}, "Statistics": {}}
        motif_tree_dict = get_motif_tree_dict()

    # OPTIMIZER
    optimizer = get_optimizer(model, config.optimizer)

    # scheduler = GradualWarmupScheduler(
    #     optimizer, multiplier=8, total_epoch=float(config.epochs), after_scheduler=None)
    # scheduler = TimmCosineLRScheduler(optimizer, t_initial=config.epochs, lr_min=1e-6)
    if config.lr_scheduler.enable:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.lr_scheduler.reset_epochs,  # First restart after 10 steps
            T_mult=config.lr_scheduler.T_mult,  # Double the period after each restart
            eta_min=config.lr_scheduler.min_lr,
            last_epoch=-1
        )

    use_amp = config.mixed_precision and device.type == 'cuda' and not config.binary_class
    amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # LOSS
    if config.pretrain:
        criterion = torch.nn.CrossEntropyLoss()
    elif config.binary_class:
        if config.nucleotide_data:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.MSELoss()

    # TRAINING
    early_stopper = EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_delta)

    logger.info(f"Starting training fold {fold}")
    start_time = time.time()
    end_time = None

    losses = {}
    best_epoch = 1
    y_true_train_best, y_pred_train_best = None, None
    y_true_val_best, y_pred_val_best = None, None
    iters = len(train_loader)
    for epoch in range(1, config.epochs + 1):
        model.train()

        running_loss = 0.0
        y_true, y_pred, tissue_ids = [], [], []
        for batch_idx, (data, target, target_bin) in enumerate(tqdm(train_loader, desc="Train Epoch")):
            data = [d.to(device) for d in data]

            if config.scale_targets and not config.binary_class:
                target = scale_y(target, **scaler_cfg)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if config.pretrain:
                    mutated_data, _, _ = get_pretrain_mask_data(epoch, copy.deepcopy(data), config, motif_cache,
                                                                motif_tree_dict)
                    output = model(mutated_data)
                    # compute combined loss, need to subtract the min token value from the target to get the correct index
                    max_batch_len = mutated_data[0].size(1)

                    loss = (
                            criterion(output[0][:, :max_batch_len, :].permute(0, 2, 1),
                                      torch.where(data[0][:, :, 0] == 0, data[0][:, :, 0],
                                                  data[0][:, :, 0] - 5).long()) +
                            criterion(output[1][:, :max_batch_len, :].permute(0, 2, 1),
                                      data[0][:, :, 1].long()) +
                            criterion(output[2][:, :max_batch_len, :].permute(0, 2, 1),
                                      torch.where(data[0][:, :, 2] == 0, data[0][:, :, 2],
                                                  data[0][:, :, 2] - 9).long()) +
                            criterion(output[3][:, :max_batch_len, :].permute(0, 2, 1),
                                      torch.where(data[0][:, :, 3] == 0, data[0][:, :, 3],
                                                  data[0][:, :, 3] - 12).long())
                    )

                    loss = loss / 4  # average loss over the 4 tasks
                    y_true.append(torch.tensor(0))  # dummy
                    y_pred.append(torch.tensor(0))  # dummy
                    tissue_ids.append(torch.tensor(0))  # dummy
                else:
                    if config.binary_class:
                        target = target_bin
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(output.squeeze().float(), target.float())

                    if config.scale_targets and not config.binary_class:
                        target = scale_y(target, **scaler_cfg, inverse=True)
                        output = scale_y(output, **scaler_cfg, inverse=True)

                    y_true.append(target.unsqueeze(1).cpu().detach().numpy())
                    y_pred.append(output.cpu().detach().numpy())
                    tissue_ids.append(data[1].cpu().detach().numpy())

            amp_scaler.scale(loss).backward()

            if config.grad_clip_norm > 0:
                if use_amp:
                    # grads are still scaled → unscale first
                    amp_scaler.unscale_(optimizer)
                # does not seem to work properly, introduces nan loss, could also be AMP though
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=config.grad_clip_norm,
                                               error_if_nonfinite=True)

            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()
            running_loss += loss.item()

            if config.lr_scheduler.enable:
                scheduler.step(epoch + batch_idx / iters)

        if config.lr_scheduler.enable:
            aim_run.track(scheduler.get_last_lr(), name='learning_rate_curr', epoch=epoch)

        train_loss = running_loss / len(train_loader)
        losses[epoch] = {"epoch": epoch, "train_loss": train_loss}
        y_true, y_pred, tissue_ids = np.vstack(y_true), np.vstack(y_pred), np.hstack(tissue_ids)

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
        if (epoch % config.val_freq == 0 or epoch % config.save_freq == 0 or epoch == config.epochs) and not config.pretrain:
            model.eval()

            val_loss = 0.0
            y_true_val, y_pred_val, tissue_ids_val = [], [], []
            with torch.no_grad():
                for data, target, target_bin in val_loader:
                    data = [d.to(device) for d in data]

                    if config.scale_targets and not config.binary_class:
                        target = scale_y(target, **scaler_cfg)

                    if config.binary_class:
                        target = target_bin
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(output.squeeze().float(), target.float())

                    if config.scale_targets and not config.binary_class:
                        target = scale_y(target, **scaler_cfg, inverse=True)
                        output = scale_y(output, **scaler_cfg, inverse=True)

                    y_true_val.append(target.unsqueeze(1).cpu().numpy())
                    y_pred_val.append(output.cpu().numpy())
                    tissue_ids_val.append(data[1].cpu().numpy())

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            losses[epoch].update({"val_loss": val_loss})
            y_true_val, y_pred_val, tissue_ids_val = np.vstack(y_true_val), np.vstack(y_pred_val), np.hstack(
                tissue_ids_val)

            logger.info(f'Validation loss: {val_loss}')
            aim_run.track(val_loss, name='val_loss', epoch=epoch)

            if config.binary_class:
                val_neg_auc = -roc_auc_score(y_true_val, y_pred_val)
                losses[epoch].update({"val_neg_auc": val_neg_auc})
                logger.info(f'Validation neg AUC:  {val_neg_auc}')
                aim_run.track(val_neg_auc, name="val_neg_auc", epoch=epoch)
                val_loss = val_neg_auc  # for early stopping in classification setting
                if val_neg_auc <= losses[best_epoch].get("val_neg_auc", 0):
                    best_epoch = epoch
                    end_time = time.time()
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                    y_true_train_best, y_pred_train_best = y_true.flatten(), y_pred.flatten()

                    evaluate(y_true_val_best, expit(y_pred_val_best), tissue_ids_val, "val", best_epoch,
                             fold, config.binary_class, os.environ["SUBPROJECT"], logger, aim_run)
            else:
                if val_loss <= losses[best_epoch].get("val_loss", np.inf):
                    best_epoch = epoch
                    end_time = time.time()
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                    y_true_train_best, y_pred_train_best = y_true.flatten(), y_pred.flatten()

                    evaluate(y_true_val_best, expit(y_pred_val_best), tissue_ids_val, "val", best_epoch,
                             fold, config.binary_class, os.environ["SUBPROJECT"], logger, aim_run)

            if early_stopper.early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                aim_run.track(1, name='early_stopping', epoch=epoch)
                break

    if not end_time:
        end_time = time.time()
    aim_run.track(1, name='training_successful')
    aim_run.track(best_epoch, name='best_epoch')

    # Save losses to a CSV file
    pd.DataFrame(losses).T.to_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))

    # Track times
    training_time = round((end_time - start_time) / 60, 4)
    logger.info(f"Training process completed. Training time for best model: {training_time} mins.")
    aim_run.track(training_time, name='training_time_min')
    aim_run.track(training_time / best_epoch, name='avg_epoch_time')

    if config.model != "mamba2":
        # Note: not possible to copute stats for mamba2
        nr_params, nr_flops = get_model_stats(config, model, device, logger)
        logger.info(f"FULL PARAMS: {sum(p.numel() for p in model.parameters())}")
        aim_run.track(nr_params, name='nr_params')
        aim_run.track(nr_flops, name='nr_flops')

    if config.clean_up_weights and not config.pretrain:
        clean_model_weights(best_epoch, fold, checkpoint_path, logger)

    logger.info(f"Weights path: {checkpoint_path}")

    if config.final_evaluation and not config.pretrain:
        metric_val = evaluate(y_true_val_best, expit(y_pred_val_best), tissue_ids_val, "val", best_epoch, fold,
                              config.binary_class, os.environ["SUBPROJECT"], logger, aim_run)
        metric_train = evaluate(y_true_train_best, expit(y_pred_train_best), tissue_ids, "train", best_epoch, fold,
                                config.binary_class, os.environ["SUBPROJECT"], logger, aim_run)
        aim_run.close()
        return {"metric_val": float(metric_val), "metric_train": float(metric_train), "best_epoch": best_epoch,
                "training_time_min": training_time}
    else:
        aim_run.close()
        return {"best_epoch": best_epoch, "training_time_min": training_time}


@discord_sender(webhook_url="https://discord.com/api/webhooks/1308890399942774946/"
                            "3UQa1CD1iNt1JRccZUxPj8ksKJzSuYcAnXMSYa8l9H4gs1DYi-t64qUR8o9-J4A1NFzS")
def train(config: DictConfig):
    logger = setup_logger()
    results = {"run": os.environ["SUBPROJECT"].replace("runs/", "", 1)}
    for fold in range(config.nr_folds):
        results[fold] = train_fold(config, logger, fold)

    if config.final_evaluation and not config.pretrain:
        metrics_val = [results[fold]["metric_val"] for fold in range(config.nr_folds)]
        metrics_train = [results[fold]["metric_train"] for fold in range(config.nr_folds)]
        results["cv_metrics_val"] = {"mean": float(np.mean(metrics_val)), "std": float(np.std(metrics_val))}
        results["cv_metric_train"] = {"mean": float(np.mean(metrics_train)), "std": float(np.std(metrics_train))}

    logger.info(results)

    return results
