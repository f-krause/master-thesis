import os
from training import train, tuning
from log.logger import setup_logger
from utils import set_log_file, set_project_path
import yaml
from box import Box

CONFIG_PATH = "config/config_template.yml"


if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = Box(yaml.safe_load(file))

    set_project_path(config)
    set_log_file(config)

    logger = setup_logger()
    logger.info(f"Project path: {os.path.join(os.environ['PROJECT_PATH'], os.environ['SUBPROJECT'])}")
    logger.info(f"Config: \n {config}")

    try:
        logger.info("TRAINING TEST")
        training.train(config)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    logger.info("COMPLETED")
    print("Log file saved at:", os.environ["LOG_FILE"])
