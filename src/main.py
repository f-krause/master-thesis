import os
import yaml
import argparse
from box import Box

from training import train, tuning
from log.logger import setup_logger
from utils import set_log_file, set_project_path

CONFIG_PATH_DEFAULT = "config/config_template.yml"

parser = argparse.ArgumentParser(prog='main.py', description='Trains DL models on mRNA data to predict PTR ratios.')

parser.add_argument('-c', '--custom_path', help="Path to config file", type=str, default=None)
parser.add_argument('-b', "--baseline", help="Use baseline config", action="store_true")
parser.add_argument('-l', "--lstm", help="Use lstm config", action="store_true")
parser.add_argument('-g', "--gru", help="Use gru config", action="store_true")
parser.add_argument('-x', "--xlstm", help="Use xlstm config", action="store_true")
parser.add_argument('-t', "--transformer", help="Use transformer config", action="store_true")
parser.add_argument('-m', "--mamba", help="Use mamba config", action="store_true")

args = parser.parse_args()

if args.custom_path:
    config_path = args.custom_path
elif args.baseline:
    config_path = "config/config_baseline.yml"
elif args.gru:
    config_path = "config/config_gru.yml"
elif args.lstm:
    config_path = "config/config_lstm.yml"
elif args.xlstm:
    config_path = "config/config_xlstm.yml"
elif args.transformer:
    config_path = "config/config_transformer.yml"
elif args.mamba:
    config_path = "config/config_mamba.yml"
else:
    config_path = CONFIG_PATH_DEFAULT


if __name__ == "__main__":
    with open(config_path, 'r') as file:
        config = Box(yaml.safe_load(file))

    set_project_path(config)
    set_log_file(config)

    logger = setup_logger()
    logger.info(f"Project path: {os.path.join(os.environ['PROJECT_PATH'], os.environ['SUBPROJECT'])}")
    logger.info(f"Config: \n {config}")

    try:
        logger.info("TRAINING TEST")
        train.train(config)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    logger.info("COMPLETED")
    print("Log file saved at:", os.environ["LOG_FILE"])
