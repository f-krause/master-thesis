import os
import argparse
from omegaconf import DictConfig
from knockknock import discord_sender

from log.logger import setup_logger
from utils import set_log_file, set_project_path, get_config, set_seed

parser = argparse.ArgumentParser(prog='main.py', description='Trains DL models on mRNA data to predict PTR ratios.')

parser.add_argument('-z', '--gpu_id', help="GPU to train on", type=int, default=None)
parser.add_argument('-s', '--custom_path', help="Path to config file", type=str, default=None)
parser.add_argument('-d', "--dummy", help="Use dummy config for test", action="store_true")
parser.add_argument('-b', "--baseline", help="Use baseline config", action="store_true")
parser.add_argument('-c', "--cnn", help="Use cnn config", action="store_true")
parser.add_argument('-l', "--lstm", help="Use lstm config", action="store_true")
parser.add_argument('-g', "--gru", help="Use gru config", action="store_true")
parser.add_argument('-x', "--xlstm", help="Use xlstm config", action="store_true")
parser.add_argument('-m', "--mamba", help="Use mamba config", action="store_true")
parser.add_argument("--jamba", help="Use jamba config", action="store_true")
parser.add_argument('-t', "--transformer", help="Use transformer config", action="store_true")
parser.add_argument("--tisnet", help="Use TISnet config", action="store_true")
parser.add_argument("--legnet", help="Use LEGnet config", action="store_true")
parser.add_argument("--ptrnet", help="Use PTRNet config", action="store_true")

args = parser.parse_args()


def main_train(config: DictConfig):
    from training import train

    set_project_path(config)
    set_log_file(config)
    set_seed(config.seed)

    logger = setup_logger()
    logger.info(f"Project path: {os.path.join(os.environ['PROJECT_PATH'], os.environ['SUBPROJECT'])}")
    logger.info(f"Config: \n {config}")

    try:
        logger.info("TRAINING STARTED")
        train.train(config)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        print("Log file saved at:", os.environ["LOG_FILE"])
        raise e

    logger.info("COMPLETED")
    print("Log file saved at:", os.environ["LOG_FILE"])


@discord_sender(webhook_url="https://discord.com/api/webhooks/1308890399942774946/"
                            "3UQa1CD1iNt1JRccZUxPj8ksKJzSuYcAnXMSYa8l9H4gs1DYi-t64qUR8o9-J4A1NFzS")
def main_param_tune(config: DictConfig):
    import optuna
    from training.param_tuning import create_objective

    set_project_path(config)
    set_seed(config.seed)

    study = optuna.create_study(
        direction='maximize' if config.binary_class else 'minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=config.optuna.n_startup_trials),
        storage=os.path.join(config.optuna.storage, config.model + '.db'),
        study_name=config.optuna.study_name,
        load_if_exists=True
    )

    objective = create_objective(config)

    study.optimize(objective, n_trials=config.optuna.n_trials, timeout=config.optuna.timeout)

    # Print the best trial information
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value: {study.best_value}')
    print(f'Best hyperparameters: {study.best_trial.params}')

    # return relevant information for discord message
    return {"best_trial": study.best_trial.number, "best_value": study.best_value,
            "best_params": study.best_trial.params}


if __name__ == "__main__":
    main_config = get_config(args)

    if main_config.param_tune:
        main_param_tune(main_config)
    else:
        main_train(main_config)
