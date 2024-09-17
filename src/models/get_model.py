import torch
from box import Box

from models.baseline.model_baseline import ModelBaseline
from models.dummy.model_dummy import ModelDummy


def get_model(config: Box, device: torch.device, logger=None):
    if config.model == "dummy":
        if logger: logger.warning("Using dummy model")
        return ModelDummy(device).to(device)
    if config.model == "baseline":
        if logger: logger.info("Using baseline model")
        return ModelBaseline(config, device).to(device)
    if config.model == "lstm":
        if logger: logger.info("Using LSTM model")
        # TODO
        raise NotImplementedError("LSTM model not implemented yet")
    if config.model == "xlstm":
        if logger: logger.info("Using xLSTM model")
        # TODO
        raise NotImplementedError("XLSTM model not implemented yet")
    if config.model == "mamba":
        if logger: logger.info("Using Mamba model")
        # TODO
        raise NotImplementedError("Mamba model not implemented yet")
    if config.model == "transformer":
        if logger: logger.info("Using Transformer model")
        # TODO
        raise NotImplementedError("Transformer model not implemented yet")
    if config.model == "best":
        if logger: logger.info("Using best model")
        # TODO
        raise NotImplementedError("Best model not implemented yet")
    else:
        raise ValueError(f"Model {config.model} not implemented! Choose from: "
                         f"dummy, baseline, lstm, xlstm, mamba, transformer, best")
