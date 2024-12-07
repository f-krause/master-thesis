import os
import aim
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, \
    root_mean_squared_error, r2_score

from utils import mkdir, log_pred_true_scatter, log_confusion_matrix, log_roc_curve


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
