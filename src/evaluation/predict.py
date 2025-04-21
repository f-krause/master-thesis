# PROBABLY UNTESTED/OUTDATED
import os
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from evaluation.evaluate import evaluate
from utils.utils import mkdir, set_project_path, get_device
from models.get_model import get_model
from log.logger import setup_logger
from data_handling.data_loader import get_train_data_loaders, get_val_data_loader, get_test_data_loader


def load_model(config: DictConfig, subproject, device, logger, full_output=False):
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], subproject, "weights")

    model = get_model(config, device, logger)
    model.eval()

    # later average models: https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/src/OpenVaccine/predict.py#L98
    for fold in range(config.nr_folds):
        # FIXME currently only returning train_loss and best_epoch of last fold loaded!
        losses = pd.read_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))
        losses = losses[~losses.checkpoint_stored.isna()]
        best_epoch = int(losses.loc[losses.val_loss.idxmin(), "epoch"])
        train_loss = losses.loc[losses.val_loss.idxmin(), "train_loss"]
        data = torch.load(os.path.join(checkpoint_path, f'checkpoint_{best_epoch}_fold-{fold}.pth.tar'),
                          weights_only=False)
        model.load_state_dict(data['state_dict'])
        # TODO allow to merge weights across folds?
    if full_output:
        return model, train_loss, best_epoch
    return model


def predict(config: DictConfig, subproject, logger, val=False, test=False, full_output=False):
    if val:
        data_loader = get_val_data_loader(config)
    elif test:
        data_loader = get_test_data_loader(config)
    else:
        _, data_loader = get_train_data_loaders(config, 1)

    device = get_device(config, logger)
    avg_model, train_loss, best_epoch = load_model(config, subproject, device, logger, full_output=True)

    # predict
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target, target_bin in tqdm(data_loader):
            data = [d.to(device) for d in data]
            if config.binary_class:
                target = target_bin
            target = target.to(device)

            output = avg_model(data)
            output, target = output.float(), target.float()

            predictions.append(output.cpu().numpy())
            targets.append(target.unsqueeze(1).cpu().numpy())

    # stack predictions
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    df = pd.DataFrame({"target": targets.squeeze(), "prediction": predictions.squeeze()})

    if full_output:
        return df, train_loss, best_epoch
    return df


if __name__ == "__main__":
    # UNTESTED
    # FIXME not working for multi-fold case! - evalute will only use last fold
    CONFIG_PATH = "config/mamba.yml"

    general_config = OmegaConf.load("config/general_codon.yml")
    model_config = OmegaConf.load(CONFIG_PATH)
    custom_config = OmegaConf.merge(general_config, model_config)

    set_project_path(custom_config)
    predictions_path = os.path.join(os.environ["PROJECT_PATH"], "runs", custom_config.subproject, "predictions")
    mkdir(predictions_path)
    os.environ["LOG_FILE"] = os.path.join(predictions_path, "log_predict.log")
    custom_logger = setup_logger()

    df, _, best_epoch = predict(custom_config, custom_config.subproject, custom_logger, full_output=True)
    evaluate(df.target, df.prediction, "train", best_epoch, custom_config.nr_folds, custom_config.binary_class,
             os.environ["SUBPROJECT"], custom_logger)
