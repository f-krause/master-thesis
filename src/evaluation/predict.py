import os
import torch
import yaml
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from utils import mkdir, set_project_path, set_log_file, get_device
from models.get_model import get_model
from log.logger import setup_logger
from data_handling.data_loader import get_data_loaders

CONFIG_PATH = "config/config_template.yml"
weights_folder = "weights"

config = OmegaConf.load(CONFIG_PATH)

set_project_path(config)
predictions_path = os.path.join(os.environ["PROJECT_PATH"], "runs", config.subproject, "predictions")
mkdir(predictions_path)
os.environ["LOG_FILE"] = os.path.join(predictions_path, "log_predict.log")
logger = setup_logger()


def load_model(config: OmegaConf, device, train_loss=False):
    checkpoint_path = os.path.join(os.environ["PROJECT_PATH"], "runs", config.subproject, weights_folder)

    model = get_model(config, device, logger)
    model.eval()

    # later average models: https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/src/OpenVaccine/predict.py#L98
    for fold in range(config.nr_folds):
        losses = pd.read_csv(os.path.join(checkpoint_path, f"losses_fold-{fold}.csv"))
        losses = losses[~losses.stored.isna()]
        best_epoch = int(losses.loc[losses.val_loss.idxmin(), "epoch"])
        train_loss = losses.loc[losses.val_loss.idxmin(), "train_loss"]
        data = torch.load(os.path.join(checkpoint_path, f'checkpoint_{best_epoch}_fold-{fold}.pth.tar'))
        model.load_state_dict(data['state_dict'])

    if train_loss:
        return model, train_loss, best_epoch
    return model


def predict():
    # get test data
    _, data_loader = get_data_loaders(config, 1)  # FIXME later test data loader

    device = get_device(config, logger)
    avg_model, train_loss, best_epoch = load_model(config, device, train_loss=True)

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
    preds_df = pd.DataFrame({"target": targets.squeeze(), "prediction": predictions.squeeze()})

    # store predictions
    preds_df.to_csv(os.path.join(predictions_path, "predictions.csv"))
    logger.info(f"Predictions stored at {predictions_path}")

    # evaluate
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    logger.info(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, RMSE_train: {train_loss}, R2: {r2}, best epoch: {best_epoch}")

    with open(os.path.join(predictions_path, "evaluation_metrics.txt"), "w") as f:
        f.write(f"MAE:        {mae}\n"
                f"MSE:        {mse}\n"
                f"RMSE:       {rmse}\n"
                f"RMSE_train: {train_loss}\n"
                f"R2:         {r2}\n"
                f"Best epoch: {best_epoch}\n")


if __name__ == "__main__":
    predict()
