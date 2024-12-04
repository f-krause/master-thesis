import os
import torch
import aim
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, roc_auc_score, \
    confusion_matrix

from utils import mkdir, set_project_path, get_device, log_pred_true_scatter, log_confusion_matrix, log_roc_curve
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


def evaluate(y_true, y_pred, dataset, best_epoch, fold, binary_class, subproject, logger, aim_tracker=None):
    logger.info("Starting evaluation")
    prediction_path = os.path.join(os.environ["PROJECT_PATH"], subproject, "predictions")
    mkdir(prediction_path)

    # Store predictions
    preds_df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    preds_df.to_csv(os.path.join(prediction_path, f"predictions_{dataset}_fold-{fold}.csv"))
    logger.info(f"Predictions stored at {prediction_path}")

    # store scatter of predictions
    img_buffer = log_pred_true_scatter(y_true, y_pred, binary_class)
    if aim_tracker: aim_tracker.track(aim.Image(img_buffer), name=f"pred_true_scatter_{dataset}")

    if binary_class:
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, [1 if target > 0.5 else 0 for target in y_pred])
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        img_buffer = log_confusion_matrix(y_true, y_pred)
        if aim_tracker: aim_tracker.track(aim.Image(img_buffer), name=f"confusion_matrix_{dataset}")

        # AUC score
        auc = roc_auc_score(y_true, y_pred)
        if aim_tracker: aim_tracker.track(auc, name=f'AUC_{dataset}')
        logger.info(f"{dataset}: AUC: {auc}, best epoch: {best_epoch}")

        with open(os.path.join(prediction_path, f"evaluation_metrics_{dataset}_fold-{fold}.txt"), "w") as f:
            f.write(f"{dataset}:\n"
                    f"AUC: {auc}\n"
                    f"Best epoch: {best_epoch}\n"
                    f"{conf_matrix}\n")

        # ROC curve
        img_buffer = log_roc_curve(y_true, y_pred)
        if aim_tracker: aim_tracker.track(aim.Image(img_buffer), name=f"roc_curve_{dataset}")
        return auc
    else:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        logger.info(f"{dataset}: MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}, best epoch: {best_epoch}")
        if aim_tracker: aim_tracker.track(r2, name=f'R2_{dataset}')

        with open(os.path.join(prediction_path, f"evaluation_metrics_{dataset}_fold-{fold}.txt"), "w") as f:
            f.write(f"{dataset}:\n"
                    f"MAE:        {mae}\n"
                    f"MSE:        {mse}\n"
                    f"RMSE:       {rmse}\n"
                    f"R2:         {r2}\n"
                    f"Best epoch: {best_epoch}\n")
        return r2


if __name__ == "__main__":
    # UNTESTED
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
    evaluate(df.target, df.prediction, "train", best_epoch, custom_config.binary_class, os.environ["SUBPROJECT"],
             custom_logger)
