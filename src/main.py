import logging
import os
import yaml
from training import training, tuning
from logs.logger import setup_logger
from utils import unique_log_file

CONFIG_PATH = "config/config_template.yml"


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    os.environ["LOG_FILE"] = config["log_file"]

    logger = setup_logger()
    logger.info("Starting training")

    try:
        # TODO try training/tuning with simple models and mockup data!
        #  training.train(config)
        logger.info("TRAINING TEST")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    logger.info("Main script execution completed")
    logging.shutdown()

    unique_log_file()
