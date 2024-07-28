import os
import yaml
from training import training, tuning
from log.logger import setup_logger
from utils import set_log_file, set_project_path

CONFIG_PATH = "config/config_template.yml"


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    set_project_path(config)
    set_log_file(config)

    logger = setup_logger()
    logger.info(f"Project path: {os.path.join(os.environ['PROJECT_PATH'], os.environ['SUBPROJECT'])}")

    try:
        logger.info("TRAINING TEST")
        training.train(config)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    logger.info("COMPLETED")
    print("Log file saved at:", os.environ["LOG_FILE"])
