import os
import datetime
import platform
import torch
import yaml
from typing import Optional
from log.logger import setup_logger
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    momentum: float


@dataclass
class TrainConfig:
    subproject: str
    project_path: Optional[str]
    log_file_path: Optional[str]
    nr_folds: int
    seed: int
    model: str
    epochs: int
    save_freq: int
    val_freq: int
    warmup: int
    batch_size: int
    num_workers: int
    optimizer: OptimizerConfig

    def __init__(self, config_file: str):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.subproject = config.get('subproject', "dev")
        self.project_path = config.get('project_path', None)
        self.log_file_path = config.get('log_file_path', None)
        self.nr_folds = config['nr_folds']
        self.seed = config['seed']
        self.model = config['model']
        self.epochs = config['epochs']
        self.save_freq = config['save_freq']
        self.val_freq = config['val_freq']
        self.warmup = config['warmup']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

        optimizer_config = config['optimizer']
        self.optimizer = OptimizerConfig(
            name=optimizer_config['name'],
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum']
        )


def mkdir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def set_log_file(config: TrainConfig):
    if not config.log_file_path:
        os.environ["LOG_FILE"] = os.path.join(os.environ["PROJECT_PATH"], config.subproject, "logs",
                                              f"log_{get_timestamp()}.log")
        mkdir(os.path.join(os.environ["PROJECT_PATH"], config.subproject, "logs"))
    else:
        os.environ["LOG_FILE"] = os.path.join(config.log_file_path, f"log_{get_timestamp()}.log")
        mkdir(config.log_file_path)


def set_project_path(config: TrainConfig):
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


# def check_file_exists(file_path, create_unique=False):
#     # FIXME DELETE?
#     logger = setup_logger()
#     if create_unique:
#         if os.path.exists(file_path):
#             logger.warning(f"File {file_path} already exists. Creating a unique file name.")
#             file_path = file_path.split(".")
#             file_path = f"{file_path[0]}_{get_timestamp()}.{file_path[1]}"
#             return file_path
#     if os.path.exists(file_path):
#         logger.error(f"File {file_path} already exists. Exiting.")
#         raise FileExistsError
#     return file_path


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
