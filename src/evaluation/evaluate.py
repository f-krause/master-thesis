import os
import aim
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

from utils.utils import mkdir, log_pred_true_scatter, log_confusion_matrix, log_roc_curve


def evaluate(y_true, y_pred, tissue_ids, dataset, best_epoch, fold, binary_class, subproject, logger, aim_tracker=None):
    logger.info("Starting evaluation")
    prediction_path = os.path.join(os.environ["PROJECT_PATH"], subproject, "predictions")
    mkdir(prediction_path)

    # Store predictions
    # FIXME should be in separate function
    preds_df = pd.DataFrame({"tissue_ids": tissue_ids, "target": y_true, "prediction": y_pred})
    preds_df.to_csv(os.path.join(prediction_path, f"predictions_{dataset}_fold-{fold}.csv"))
    logger.info(f"Predictions stored at {prediction_path}")

    # store scatter of predictions
    for limits in [(0, 10), (10, 20), (20, 29), (0, 29)]:
        img_buffer = log_pred_true_scatter(y_true, y_pred, tissue_ids, limits, binary_class)
        if aim_tracker: aim_tracker.track(aim.Image(img_buffer), name=f"pred_true_scatter_{dataset}_{limits[1]}")

    if binary_class:
        Y = np.where(np.array(y_pred) > 0.5, 1, 0)
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, Y)
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        img_buffer = log_confusion_matrix(y_true, y_pred)
        if aim_tracker: aim_tracker.track(aim.Image(img_buffer), name=f"confusion_matrix_{dataset}")

        logger.info(f"best epoch: {best_epoch}")

        # AUC score
        auc = roc_auc_score(y_true, y_pred)
        if aim_tracker: aim_tracker.track(auc, name=f'AUC_{dataset}')
        logger.info(f"{dataset}: AUC: {auc}")

        # Accuracy score
        accuracy = accuracy_score(y_true, Y)
        if aim_tracker: aim_tracker.track(accuracy, name=f'Accuracy_{dataset}')
        logger.info(f"{dataset}: Accuracy: {accuracy}")

        # Precision score
        precision = precision_score(y_true, Y)
        if aim_tracker: aim_tracker.track(precision, name=f'Precision_{dataset}')
        logger.info(f"{dataset}: Precision: {precision}")

        # Recall score
        recall = recall_score(y_true, Y)
        if aim_tracker: aim_tracker.track(recall, name=f'Recall_{dataset}')
        logger.info(f"{dataset}: Recall: {recall}")

        # F1 score
        f1 = f1_score(y_true, Y)
        if aim_tracker: aim_tracker.track(f1, name=f'F1_{dataset}')
        logger.info(f"{dataset}: F1 Score: {f1}")

        # Write metrics to file
        with open(os.path.join(prediction_path, f"evaluation_metrics_{dataset}_fold-{fold}.txt"), "w") as f:
            f.write(f"{dataset}:\n"
                    f"AUC: {auc}\n"
                    f"Accuracy: {accuracy}\n"
                    f"Precision: {precision}\n"
                    f"Recall: {recall}\n"
                    f"F1: {f1}\n"
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
