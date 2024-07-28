import os
import logging


def setup_logger():
    log_file = os.environ.get("LOG_FILE")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
