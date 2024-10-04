import os
import datetime
import platform
import torch
from log.logger import setup_logger
from omegaconf import OmegaConf


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def set_log_file(config: OmegaConf, predict=False):
    if predict:
        log_file_name = f"log_predict_{get_timestamp()}.log"
    else:
        log_file_name = f"log_{get_timestamp()}.log"

    if not config.log_file_path:
        os.environ["LOG_FILE"] = os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "logs", log_file_name)
        mkdir(os.path.join(os.environ["PROJECT_PATH"], os.environ["SUBPROJECT"], "logs"))
    else:
        os.environ["LOG_FILE"] = os.path.join(config.log_file_path, log_file_name)
        mkdir(config.log_file_path)


def set_project_path(config: OmegaConf):
    if config.project_path:
        project_path = config.project_path
    elif platform.node() == "Felix-PC":
        project_path = r"C:\Users\Felix\code\uni\UniVie\master-thesis-data"
    elif platform.node() == "TGA-NB-060":
        project_path = r"C:\Users\felix.krause\code\uni\master-thesis-data"
    elif platform.node() == "rey" or platform.node() == "jyn":
        project_path = r"/export/share/krausef99dm"
    else:
        raise ValueError("Unknown platform. Please specify project path in config file.")

    os.environ["PROJECT_PATH"] = project_path

    subproject_path = os.path.join(project_path, "runs",  config.subproject)
    if os.path.isdir(subproject_path):
        os.environ["SUBPROJECT"] = os.path.join("runs", config.subproject + "_" + get_timestamp())
    else:
        os.environ["SUBPROJECT"] = os.path.join("runs", config.subproject)

    return project_path


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def get_config(args, default="config/baseline.yml"):
    if args.custom_path:
        return args.custom_path
    if args.dummy:
        return "config/dummy.yml"
    elif args.baseline:
        return "config/baseline.yml"
    elif args.gru:
        return "config/gru.yml"
    elif args.lstm:
        return "config/lstm.yml"
    elif args.xlstm:
        return "config/xlstm.yml"
    elif args.mamba:
        return "config/mamba.yml"
    elif args.jamba:
        return "config/jamba.yml"
    elif args.transformer:
        return "config/transformer.yml"
    else:
        raise ValueError("No config file specified.")


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


def get_device(config: OmegaConf, logger=None):
    if logger:
        logger.info(f"Devices found for training: "
                    f"{[(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if logger:
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info("Using GPU: " + str(config.gpu_id))
    return device
