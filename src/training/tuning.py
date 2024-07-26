# TODO https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7
import optuna
import yaml
import datetime
from ..training.training import train


def objective(trial):
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
    train(config)


def store_best_config(study_obj):
    print("Storing best hyperparameters")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    with open(f'../config/config_{timestamp}.yml', 'w') as f:
        yaml.dump(study_obj.best_params, f)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
store_best_config(study)
print(f"Best hyperparameters: {study.best_params}")
