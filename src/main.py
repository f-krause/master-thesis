import os
import yaml
import argparse
from box import Box

from training import train, tuning
from log.logger import setup_logger
from utils import set_log_file, set_project_path, get_config

parser = argparse.ArgumentParser(prog='main.py', description='Trains DL models on mRNA data to predict PTR ratios.')

parser.add_argument('-c', '--custom_path', help="Path to config file", type=str, default=None)
parser.add_argument('-d', "--dummy", help="Use dummy config for test", action="store_true")
parser.add_argument('-b', "--baseline", help="Use baseline config", action="store_true")
parser.add_argument('-l', "--lstm", help="Use lstm config", action="store_true")
parser.add_argument('-g', "--gru", help="Use gru config", action="store_true")
parser.add_argument('-x', "--xlstm", help="Use xlstm config", action="store_true")
parser.add_argument('-t', "--transformer", help="Use transformer config", action="store_true")
parser.add_argument('-m', "--mamba", help="Use mamba config", action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    config_path = get_config(args)

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
