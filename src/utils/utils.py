import os
import datetime
import platform
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from fvcore.nn import FlopCountAnalysis, flop_count_table

from log.logger import setup_logger
from utils.knowledge_db import CODON_MAP_DNA
from data_handling.data_utils import get_one_hot_tensors


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
    elif config.pretrain:
        path_components.append("pretrain")
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
    elif args.transformer:
        config_path = "config/transformer.yml"
    elif args.legnet:
        config_path = "config/LegNet.yml"
    elif args.ribonn:
        config_path = "config/RiboNN.yml"
    elif args.ptrnet:
        config_path = "config/PTRnet.yml"
    else:
        raise ValueError("No config file specified.")

    model_config = OmegaConf.load(config_path)

    if args.ptrnet:  # TODO add more cases
        # Config for nucleotide sequence based models
        general_config = OmegaConf.load("config/general_nucleotide.yml")
    else:
        # Config for codon based models
        general_config = OmegaConf.load("config/general_codon.yml")

    config = OmegaConf.merge(general_config, model_config)  # with model_config on the right, overwrites general

    if args.gpu_id is not None:
        OmegaConf.update(config, "gpu_id", args.gpu_id)

    config = check_config(config)

    return config


def check_config(config: DictConfig):
    if config.nr_folds > 1 and not config.concat_train_val:
        raise ValueError("If running cv, concat_train_val should be True to run on train + val data.")
    if config.nr_folds > 1 and config.evaluate_on_test:
        raise ValueError("If running cv, evaluate_on_test should be False to evaluate on validation fold.")
    if config.scale_targets and config.binary_class:
        raise ValueError("If using target scaling, binary classification should be False.")
    if not config.get("nucleotide_data", False):
        # Add nucleotide data specific params to config automatically for codon case
        config.nucleotide_data = False
        config.seq_encoding = "embedding"
        config.align_aug = False
        config.pretrain = False
    if config.align_aug and config.random_reverse:
        raise ValueError("If using alignment augmentation, random reverse should be False.")
    if config.seq_encoding not in ["embedding", "ohe"]:  # legacy: word2vec
        raise ValueError(f"Unknown sequence encoding: {config.seq_encoding}. Choose from embedding, ohe, word2vec.")
    if config.align_aug:
        # Set new max_seq_length for alignment augmentation
        config.max_seq_length = config.max_seq_length_aug_alignment
    if config.pretrain and config.model != "ptrnet":
        raise ValueError("Pretraining is only supported for PTRnet model.")
    if config.pretrain and config.random_reverse:
        raise ValueError("Random reverse is not supported for pretraining.")
    if config.pretrain:
        # If pretraining, force model to not be in binary classification mode
        OmegaConf.update(config, "binary_class", False)
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


def clean_model_weights(best_epoch, fold, checkpoint_path, logger):
    logger.info(f"Starting weights cleanup")
    files = os.listdir(checkpoint_path)
    if f"checkpoint_{best_epoch}_fold-{fold}" not in str(files):
        raise FileNotFoundError(f"Could not find the best epoch weights: checkpoint_{best_epoch}_fold-{fold}")
    # only keep files ending on correct fold
    files = [file for file in files if file.split(".")[0].endswith(str(fold)) and file.startswith("checkpoint")]
    for file in files:
        if f"checkpoint_{best_epoch}_fold-{fold}" not in file and file.endswith(".tar"):
            os.remove(os.path.join(checkpoint_path, file))
    logger.info(f"Removed all checkpoint weights except the best one: checkpoint_{best_epoch}_fold-{fold}")


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
    if config.nucleotide_data:
        rna_data, seq_lengths = [], []
        for i in range(config.batch_size):
            length = torch.randint(100, config.max_seq_length_aug_alignment + 1, (1,)).item()
            # Generate each column with its own integer range
            col0 = torch.randint(low=6, high=10, size=(length,))  # values in [6,9] - seq ohe
            col1 = torch.randint(low=1, high=6, size=(length,))  # values in [1,5] - coding area
            col2 = torch.randint(low=10, high=13, size=(length,))  # values in [10,12] - loop type pred
            col3 = torch.randint(low=13, high=20, size=(length,))  # values in [13,19] - sec structure pred

            seq = torch.stack([col0, col1, col2, col3], dim=-1)
            rna_data.append(seq)
            seq_lengths.append(length)

        tissue_ids = torch.randint(29, (config.batch_size,))  # tissue_ids (B)

        if config.seq_encoding.lower() == "ohe":
            rna_data = get_one_hot_tensors(config, rna_data, tissue_ids)

        sample_input = [
            torch.nn.utils.rnn.pad_sequence(rna_data, batch_first=True),  # rna_data_padded (B x N x D)
            tissue_ids,  # tissue_ids (B)
            torch.tensor(seq_lengths, dtype=torch.int64),  # seq_lengths (B)
            torch.randn(config.batch_size, len(CODON_MAP_DNA)),  # frequency_features (B x 64)
        ]

    else:
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


def fit_target_scaler(t: torch.Tensor):
    return t.mean().item(), t.std().item()


def scale_y(t: torch.Tensor, mu: float, sigma: float, inverse=False):
    return (t * sigma + mu) if inverse else (t - mu) / sigma
