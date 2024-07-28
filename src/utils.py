import os
import datetime
import platform
import torch
from log.logger import setup_logger


def mkdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def set_log_file(config):
    if not config["log_file_path"]:
        os.environ["LOG_FILE"] = os.path.join(os.environ["PROJECT_PATH"], config["subproject"], "logs",
                                              f"log_{get_timestamp()}.log")
        mkdir(os.path.join(os.environ["PROJECT_PATH"], config["subproject"], "logs"))
    else:
        os.environ["LOG_FILE"] = os.path.join(config["log_file_path"], f"log_{get_timestamp()}.log")
        mkdir(config["log_file_path"])


def set_project_path(config):
    if config["project_path"]:
        project_path = config["project_path"]
    elif platform.node() == "Felix-PC":
        project_path = r"C:\Users\Felix\code\uni\UniVie\master-thesis-data"
    else:
        project_path = "krausef99dm_thesis"

    os.environ["PROJECT_PATH"] = project_path
    os.environ["SUBPROJECT"] = config["subproject"]
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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename):
    return torch.load(filename)
