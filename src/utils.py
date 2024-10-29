import os
import io
import datetime
import platform
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from log.logger import setup_logger
from omegaconf import DictConfig
from fvcore.nn import FlopCountAnalysis, flop_count_table


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def set_log_file(config: DictConfig, predict=False):
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


def set_project_path(config: DictConfig):
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


def get_config(args):
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


def get_device(config: DictConfig, logger=None):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)  # removed as not working
    if logger:
        logger.info(f"Devices found for training: "
                    f"{[(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_id}")
    else:
        device = "cpu"

    if logger:
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info("Using GPU: " + str(config.gpu_id))
    return device


def set_seed(seed):
    """ Set seed for reproducibility """
    random.seed(seed + 1)
    np.random.seed(seed + 2)
    torch.manual_seed(seed + 3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 4)
        torch.cuda.manual_seed_all(seed + 5)


def get_model_stats(config: DictConfig, model, device, logger):
    logger.info("Computing model statistics, assuming input is max codon sequence and tissue type (only).")
    sample_input = [
        # rna_data_padded (batch_size x max_seq_length)
        torch.nn.utils.rnn.pad_sequence(torch.randint(1, 64, (1, config.max_seq_length)), batch_first=True),
        torch.randint(29, (1,)),  # tissue_ids (batch_size x 1)
        torch.tensor([config.max_seq_length], dtype=torch.int64)  # seq_lengths (batch_size x 1)
    ]

    sample_input = [x.to(device) for x in sample_input]
    flops = FlopCountAnalysis(model, sample_input)
    nr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_nr = round(flops.total())

    logger.info(f"Model: Total parameters: {nr_params}")
    logger.info(f"Model: FLOPs: {flops_nr}")
    logger.info(f"Model: FLOPs table: \n{flop_count_table(flops)}")


def log_pred_true_scatter(y_true, y_pred):
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('y_true vs y_pred')

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--')

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)
