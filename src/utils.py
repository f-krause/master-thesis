import os
import datetime
import platform
import torch
from log.logger import setup_logger
from box import Box


TISSUES = ['Adrenal', 'Appendices', 'Brain', 'Colon', 'Duodenum', 'Uterus',
       'Esophagus', 'Fallopiantube', 'Fat', 'Gallbladder', 'Heart', 'Kidney',
       'Liver', 'Lung', 'Lymphnode', 'Ovary', 'Pancreas', 'Placenta',
       'Prostate', 'Rectum', 'Salivarygland', 'Smallintestine', 'Smoothmuscle',
       'Spleen', 'Stomach', 'Testis', 'Thyroid', 'Tonsil', 'Urinarybladder']


def mkdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def set_log_file(config: Box):
    if not config.log_file_path:
        os.environ["LOG_FILE"] = os.path.join(os.environ["PROJECT_PATH"], config.subproject, "logs",
                                              f"log_{get_timestamp()}.log")
        mkdir(os.path.join(os.environ["PROJECT_PATH"], config.subproject, "logs"))
    else:
        os.environ["LOG_FILE"] = os.path.join(config.log_file_path, f"log_{get_timestamp()}.log")
        mkdir(config.log_file_path)


def set_project_path(config: Box):
    if config.project_path:
        project_path = config.project_path
    elif platform.node() == "Felix-PC":
        project_path = r"C:\Users\Felix\code\uni\UniVie\master-thesis-data"
    elif platform.node() == "TGA-NB-060":
        project_path = r"C:\Users\felix.krause\code\uni\master-thesis-data"
    elif platform.node() == "rey":
        project_path = r"/mnt/data/krausef99dm_thesis"
    else:
        raise ValueError("Unknown platform. Please specify project path in config file.")

    os.environ["PROJECT_PATH"] = project_path
    os.environ["SUBPROJECT"] = config.subproject

    return project_path


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def check_path_exists(file_path, create_unique=False):
    logger = setup_logger()

    if os.path.isdir(file_path) and os.listdir(file_path):
        if create_unique:
            logger.warning(f"Overwriting risk for path {file_path}. Creating a unique path.")
            timestamp = os.environ["LOG_FILE"].split("_")[-1].split(".")[0]
            file_path = f"{file_path}_{timestamp}"
            return file_path
        else:
            logger.error(f"Path {file_path} already exists. Exiting.")
            raise FileExistsError

    return file_path


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(filename):
    return torch.load(filename)
