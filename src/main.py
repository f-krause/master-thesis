import os
import argparse
from omegaconf import OmegaConf

from training import train, tuning
from log.logger import setup_logger
from utils import set_log_file, set_project_path, get_config, set_seed

parser = argparse.ArgumentParser(prog='main.py', description='Trains DL models on mRNA data to predict PTR ratios.')

parser.add_argument('-s', '--custom_path', help="Path to config file", type=str, default=None)
parser.add_argument('-d', "--dummy", help="Use dummy config for test", action="store_true")
parser.add_argument('-b', "--baseline", help="Use baseline config", action="store_true")
parser.add_argument('-c', "--cnn", help="Use cnn config", action="store_true")
parser.add_argument('-l', "--lstm", help="Use lstm config", action="store_true")
parser.add_argument('-g', "--gru", help="Use gru config", action="store_true")
parser.add_argument('-x', "--xlstm", help="Use xlstm config", action="store_true")
parser.add_argument('-m', "--mamba", help="Use mamba config", action="store_true")
parser.add_argument('-j', "--jamba", help="Use jamba config", action="store_true")
parser.add_argument('-t', "--transformer", help="Use transformer config", action="store_true")
parser.add_argument('-p', "--ptrnet", help="Use PTRNet config", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    config_path = get_config(args)

    if args.ptrnet:
        general_config = OmegaConf.load("config/general_base.yml")
    else:
        general_config = OmegaConf.load("config/general_codon.yml")

    model_config = OmegaConf.load(config_path)
    config = OmegaConf.merge(general_config, model_config)

    set_project_path(config)
    set_log_file(config)
    set_seed(config.seed)

    logger = setup_logger()
    logger.info(f"Project path: {os.path.join(os.environ['PROJECT_PATH'], os.environ['SUBPROJECT'])}")
    logger.info(f"Config: \n {config}")

    if config.save_freq % config.val_freq != 0:
        raise ValueError(f"save_freq ({config.save_freq}) should be a multiple of val_freq ({config.val_freq})")

    try:
        logger.info("TRAINING STARTED")
        train.train(config)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        print("Log file saved at:", os.environ["LOG_FILE"])
        raise e

    logger.info("COMPLETED")
    print("Log file saved at:", os.environ["LOG_FILE"])
