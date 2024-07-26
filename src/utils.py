import torch
import datetime
import os
from logs.logger import setup_logger


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def check_file_exists(file_path, create_unique=False):
    logger = setup_logger()
    if create_unique:
        if os.path.exists(file_path):
            logger.warning(f"File {file_path} already exists. Creating a unique file name.")
            file_path = file_path.split(".")
            file_path = f"{file_path[0]}_{get_timestamp()}.{file_path[1]}"
            return file_path
    if os.path.exists(file_path):
        logger.error(f"File {file_path} already exists. Exiting.")
        raise FileExistsError
    return file_path


def unique_log_file():
    log_file_orig = os.environ.get("LOG_FILE")

    log_file = log_file_orig.split(".")
    log_file = f"{log_file[0]}_{get_timestamp()}.{log_file[1]}"

    os.rename(log_file_orig, log_file)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename):
    return torch.load(filename)
