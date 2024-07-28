import os
import datetime
import platform
import torch
from logs.logger import setup_logger


def mkdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def set_log_file(config):
    if not config["log_file_path"]:
        os.environ["LOG_FILE"] = os.path.join("logs", config["subproject"], "logging.log")
        mkdir(os.path.join("logs", config["subproject"]))
    else:
        os.environ["LOG_FILE"] = os.path.join(config["log_file_path"], "logging.log")
        mkdir(config["log_file_path"])


def set_project_path(config):
    logger = setup_logger()
    if config["project_path"]:
        project_path = config["project_path"]
    elif platform.node() == "Felix-PC":
        project_path = r"C:\Users\Felix\code\uni\UniVie\master-thesis-data"
    else:
        project_path = "krausef99dm_thesis"

    os.environ["PROJECT_PATH"] = project_path
    os.environ["SUBPROJECT"] = config["subproject"]
    logger.info(f"Project path set to {project_path}")
    logger.info(f"Subproject set to {config['subproject']}")
    return project_path


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
