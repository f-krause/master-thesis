import os
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from utils import mkdir, set_project_path, get_device
from models.get_model import get_model
from log.logger import setup_logger
from data_handling.data_loader import get_train_data_loaders, get_val_data_loader, get_test_data_loader


def clean_model_weights(best_epoch, fold, checkpoint_path, logger):
    # Remove all weights except the best one
    for file in os.listdir(checkpoint_path):
        if f"checkpoint_{best_epoch}_fold-{fold}" not in file:
            os.remove(os.path.join(checkpoint_path, file))
    logger.info(f"Removed all checkpoint weights except the best one: checkpoint_{best_epoch}_fold-{fold}")


def load_model(config: DictConfig, subproject, device, logger, full_output=False):
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], subproject, "weights")

    model = get_model(config, device, logger)
    model.eval()

    # later average models: https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/src/OpenVaccine/predict.py#L98
    for fold in range(config.nr_folds):
        # FIXME currently only returning last train_loss and best_epoch
        losses = pd.read_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))
        losses = losses[~losses.checkpoint_stored.isna()]
        best_epoch = int(losses.loc[losses.val_loss.idxmin(), "epoch"])
        train_loss = losses.loc[losses.val_loss.idxmin(), "train_loss"]
        data = torch.load(os.path.join(checkpoint_path, f'checkpoint_{best_epoch}_fold-{fold}.pth.tar'),
                          weights_only=False)
        if config.clean_up_weights:
            clean_model_weights(best_epoch, fold, checkpoint_path, logger)
        model.load_state_dict(data['state_dict'])

    if full_output:
        return model, train_loss, best_epoch
    return model


def evaluate(target, prediction):
    mae = mean_absolute_error(target, prediction)
    mse = mean_squared_error(target, prediction)
    rmse = root_mean_squared_error(target, prediction)
    r2 = r2_score(target, prediction)
    return mae, mse, rmse, r2


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
        for data, target in tqdm(data_loader):
            data = [d.to(device) for d in data]
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


def predict_and_evaluate(config: DictConfig, subproject, logger):
    prediction_path = os.path.join(os.environ["PROJECT_PATH"], subproject, "predictions")
    mkdir(prediction_path)

    preds_df, train_loss, best_epoch = predict(config, subproject, logger, full_output=True)
    preds_df.to_csv(os.path.join(prediction_path, "predictions.csv"))
    logger.info(f"Predictions stored at {prediction_path}")

    mae, mse, rmse, r2 = evaluate(preds_df.target, preds_df.prediction)

    logger.info(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, RMSE_train: {train_loss}, R2: {r2}, best epoch: {best_epoch}")

    with open(os.path.join(prediction_path, "evaluation_metrics.txt"), "w") as f:
        f.write(f"MAE:        {mae}\n"
                f"MSE:        {mse}\n"
                f"RMSE:       {rmse}\n"
                f"RMSE_train: {train_loss}\n"
                f"R2:         {r2}\n"
                f"Best epoch: {best_epoch}\n")


if __name__ == "__main__":
    CONFIG_PATH = "config/mamba.yml"

    general_config = OmegaConf.load("config/general.yml")
    model_config = OmegaConf.load(CONFIG_PATH)
    custom_config = OmegaConf.merge(general_config, model_config)

    set_project_path(custom_config)
    predictions_path = os.path.join(os.environ["PROJECT_PATH"], "runs", custom_config.subproject, "predictions")
    mkdir(predictions_path)
    os.environ["LOG_FILE"] = os.path.join(predictions_path, "log_predict.log")
    custom_logger = setup_logger()

    predict_and_evaluate(custom_config, os.path.join("runs", custom_config.subproject), custom_logger)
