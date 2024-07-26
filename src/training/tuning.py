# TODO https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7
import optuna
import yaml
from training.training import train
from logs.logger import setup_logger
from utils import get_timestamp

# TODO add arguments to define values like n_trials?


def tuning_objective(trial):
    logger = setup_logger()
    logger.info("Starting hyperparameter tuning")

    config = {
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'num_workers': trial.suggest_int('num_workers', 2, 8),
        'epochs': 10,
        'save_freq': 5,
        'optimizer': {
            'name': trial.suggest_categorical('optimizer', ['adam', 'sgd']),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
            'momentum': 0.9  # Only for SGD
        }
    }
    try:
        train(config)
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise e

    logger.info("Hyperparameter tuning completed")


def store_best_config(study_obj):
    print("Storing best hyperparameters")
    with open(f'../config/config_{get_timestamp()}.yml', 'w') as f:
        yaml.dump(study_obj.best_params, f)


def run_tuning():
    logger = setup_logger()
    study = optuna.create_study(direction='minimize')
    study.optimize(tuning_objective, n_trials=10)
    store_best_config(study)
    logger.info(f"Best hyperparameters: {study.best_params}")
