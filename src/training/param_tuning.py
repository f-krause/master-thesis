import torch
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from omegaconf import OmegaConf, DictConfig

from utils.utils import get_device, set_seed
from models.get_model import get_model
from data_handling.data_loader import get_train_data_loaders
from training.optimizer import get_optimizer
from training.early_stopper import EarlyStopper
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def train_tune_fold(config: DictConfig, train_loader, val_loader, trial):
    device = get_device(config)
    model = get_model(config, device)
    optimizer = get_optimizer(model, config.optimizer)
    # criterion = torch.nn.BCELoss() if config.binary_class else torch.nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss() if config.binary_class else torch.nn.MSELoss()
    early_stopper = EarlyStopper(patience=config.early_stopper_patience, min_delta=config.early_stopper_delta)

    if config.lr_scheduler.enable:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.lr_scheduler.reset_epochs,
                                                T_mult=config.lr_scheduler.T_mult,
                                                eta_min=config.lr_scheduler.min_lr, last_epoch=-1)

    losses = {}

    best_epoch = 1
    y_true_val_best, y_pred_val_best = None, None
    iters = len(train_loader)
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

            # FOR DEBUGGING - why training fails
            # temp_output = output.cpu().detach().numpy()
            # if np.isnan(temp_output).any():
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             trial.set_user_attr(f"grad_{name}_max", param.grad.max().item())
            #             trial.set_user_attr(f"grad_{name}_min", param.grad.min().item())
            #         else:
            #             trial.set_user_attr(f"grad_{name}_max", None)
            #             trial.set_user_attr(f"grad_{name}_min", None)
            #     trial.set_user_attr("data input to model:", str(data))
            #     trial.set_user_attr("temp_output contains NaN:", str(temp_output))
            #     print(f"WARNING: temp_output contains NaN: {temp_output}")
            #     raise RuntimeError("temp_output contains NaN")
            # if (temp_output > 1).any() or (temp_output < 0).any():
            #     trial.set_user_attr("probs_out_of_bounds", str(temp_output))
            #     print(f"WARNING: probs out of bounds: min={temp_output.min().item():.2f}, max={temp_output.max().item():.2f}")

            loss = criterion(output.squeeze().float(), target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y_true.append(target.unsqueeze(1).cpu().detach().numpy())
            y_pred.append(output.cpu().detach().numpy())

            if config.lr_scheduler.enable:
                scheduler.step(epoch + batch_idx / iters)

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
                val_auc = roc_auc_score(y_true_val, y_pred_val)
                losses[epoch].update({"val_auc": val_auc})
                if val_auc > losses[best_epoch].get("val_auc", 0):
                    best_epoch = epoch
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                trial.report(val_auc, epoch)
                val_loss = -val_auc  # for early stopping in classification setting
            else:
                if val_loss < losses[best_epoch].get("val_loss", np.inf):
                    best_epoch = epoch
                    y_true_val_best, y_pred_val_best = y_true_val.flatten(), y_pred_val.flatten()
                trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stopper.early_stop(val_loss):
                break

    if y_true_val_best is None and y_pred_val_best is None:
        raise optuna.exceptions.TrialPruned()
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
        except RuntimeError as e:  # skip CUDA out of memory
            print(e)
            trial.set_user_attr("failure_reason", str(e))
            raise optuna.exceptions.TrialPruned()
        return score

    return objective


def set_trial_parameters(trial, config):
    # load the hyperparameters from the yml file with OmegaConfig
    params_general = OmegaConf.load('config/param_tuning/general_param.yml')

    # set the hyperparameters for the trial
    # config.batch_size = trial.suggest_categorical('batch_size', params_general.batch_size)
    config.predictor_hidden = trial.suggest_categorical('predictor_hidden', params_general.predictor_hidden)
    # config.random_reverse = trial.suggest_categorical('random_reverse', params_general.random_reverse)
    config.lr = trial.suggest_float('lr', params_general.lr.min, params_general.lr.max, log=True)

    if config.model == "ptrnet":
        config.predictor_dropout = trial.suggest_categorical('predictor_dropout', params_general.predictor_dropout)
        config.weight_decay = trial.suggest_float('weight_decay', params_general.weight_decay.min,
                                                  params_general.weight_decay.max, log=True)
        if params_general.lr_scheduler_enable:
            config.lr_scheduler.enable = True
            config.lr_scheduler.reset_epochs = trial.suggest_categorical('reset_epochs', params_general.reset_epochs)
            config.lr_scheduler.T_mult = trial.suggest_categorical('T_mult', params_general.T_mult)
        else:
            config.lr_scheduler.enable = False

    try:
        params_model = OmegaConf.load(f'config/param_tuning/{config.model}_param.yml')
    except FileNotFoundError:
        raise Exception(f"No yml file for {config.model} found at config/param_tuning for hyperparameter tuning.")

    for key, values in params_model.items():
        if config.model in ["RiboNN", "ptrnet"] and key == "grad_clip_norm":
            # Edge case handling
            continue
        config[key] = trial.suggest_categorical(key, values)

    if config.model == "cnn":
        if config.num_kernels_conv1 < config.num_kernels_conv2:
            config.num_kernels_conv1 = config.num_kernels_conv2 * 2
        if config.max_pool1 < config.max_pool2:
            config.max_pool1 = config.max_pool2 + 10

    if config.model == "xlstm":
        if config.num_blocks == 5 or config.num_blocks == 6:
            config.slstm_at = "all"
        else:
            config.slstm_at = []

    if config.model in ["RiboNN", "ptrnet"]:
        config.grad_clip_norm = trial.suggest_float("grad_clip_norm", params_model.grad_clip_norm.min,
                                                    params_model.grad_clip_norm.max, log=True)

    if config.model == "ptrnet":
        if config.seq_encoding == "ohe":
            config.seq_only = False

        if config.seq_encoding == "embedding":
            if config.concat_tissue_feature:
                config.dim_embedding_token = trial.suggest_categorical('dim_embedding', params_general.dim_embedding)
                config.dim_embedding_tissue = trial.suggest_categorical('dim_embedding_tissue',
                                                                        params_general.dim_embedding_tissue)
            else:
                config.dim_embedding_token = trial.suggest_categorical('dim_embedding', params_general.dim_embedding)
                config.dim_embedding_tissue = config.dim_embedding_token
        elif config.seq_encoding == "ohe":
            # only need embedding for tissue
            config.dim_embedding_tissue = trial.suggest_categorical('dim_embedding_tissue',
                                                                    params_general.dim_embedding_tissue)
            config.concat_tissue_feature = True
        else:
            raise Exception(f"Unsupported seq_encoding {config.seq_encoding} for ptrnet model.")
    else:
        config.dim_embedding_token = trial.suggest_categorical('dim_embedding', params_general.dim_embedding)
        config.dim_embedding_tissue = config.dim_embedding_token  # needs to be of same size as summed

    if config.model == "mamba2":
        if (config.dim_embedding_token * config.expand / config.head_dim) % 8 != 0:
            print("WARNING: Infeasible combination of parameters")
            raise optuna.exceptions.TrialPruned()
