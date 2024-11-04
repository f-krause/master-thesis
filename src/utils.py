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
from omegaconf import DictConfig, OmegaConf
from fvcore.nn import FlopCountAnalysis, flop_count_table
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay


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


def get_run_path(config, project_path, runs_folder="runs"):
    path_components = [runs_folder]

    if "dev" in config.train_data_file or config.dev:
        path_components.append("dev")

    if config.binary_class:
        path_components.append("binary")
    else:
        path_components.append("regr")

    model_part = config.model
    path_components.append(model_part)

    run_path = os.path.join(*path_components)

    if config.frequency_features:
        model_part += "_freq"

    version = 1
    while os.path.isdir(os.path.join(project_path, run_path, f"{version}_{model_part}")):
        version += 1

    return os.path.join(run_path, f"{version}_{model_part}")


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

    subproject_path = get_run_path(config, project_path, runs_folder="runs")
    os.environ["SUBPROJECT"] = subproject_path

    return project_path


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def get_config(args):
    if args.custom_path:
        config_path = args.custom_path
    elif args.dummy:
        config_path = "config/dummy.yml"
    elif args.baseline:
        config_path = "config/baseline.yml"
    elif args.cnn:
        config_path = "config/cnn.yml"
    elif args.gru:
        config_path = "config/gru.yml"
    elif args.lstm:
        config_path = "config/lstm.yml"
    elif args.xlstm:
        config_path = "config/xlstm.yml"
    elif args.mamba:
        config_path = "config/mamba.yml"
    elif args.jamba:
        config_path = "config/jamba.yml"
    elif args.transformer:
        config_path = "config/transformer.yml"
    else:
        raise ValueError("No config file specified.")

    model_config = OmegaConf.load(config_path)

    if args.ptrnet:  # TODO add more cases
        general_config = OmegaConf.load("config/general_base.yml")
    else:
        general_config = OmegaConf.load("config/general_codon.yml")

    config = OmegaConf.merge(general_config, model_config)

    if args.gpu_id is not None:
        OmegaConf.update(config, "gpu_id", args.gpu_id)

    return config


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

    return nr_params, flops_nr


def log_pred_true_scatter(y_true, y_pred, binary_class=False):
    plt.figure(figsize=(10, 10))
    if binary_class:
        # add small random variation to scatter plot to make it more readable
        y_true = y_true + np.random.normal(0, 0.02, len(y_true))
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


def log_confusion_matrix(y_true, y_pred):
    y_pred = [1 if target > 0.5 else 0 for target in y_pred]  # make target binary, tau = 0.5

    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)


def log_roc_curve(y_true, y_pred):
    plt.figure(figsize=(10, 10))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='estimator')
    disp.plot()
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')

    # Save the plot to an in-memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return Image.open(buffer)
