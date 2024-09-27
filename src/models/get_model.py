import torch
from box import Box

from models.baseline.baseline import ModelBaseline
from models.dummy.dummy import ModelDummy
from models.rnn.rnn import ModelRNN
from models.xlstm.xlstm import ModelXLSTM


def get_model(config: Box, device: torch.device, logger=None):
    if config.model == "dummy":
        if logger: logger.warning("Using dummy model")
        return ModelDummy(device).to(device)
    elif config.model == "baseline":
        if logger: logger.info("Using baseline model")
        return ModelBaseline(config, device).to(device)
    elif config.model == "gru":
        if logger: logger.info("Using GRU model")
        return ModelRNN(config, device, model="gru").to(device)
    elif config.model == "lstm":
        if logger: logger.info("Using LSTM model")
        return ModelRNN(config, device, model="lstm").to(device)
    elif config.model == "xlstm":
        if logger: logger.info("Using xLSTM model")
        return ModelXLSTM(config, device).to(device)
    elif config.model == "mamba":
        if logger: logger.info("Using Mamba model")
        # TODO
        raise NotImplementedError("Mamba model not implemented yet")
    elif config.model == "transformer":
        if logger: logger.info("Using Transformer model")
        # TODO
        raise NotImplementedError("Transformer model not implemented yet")
    elif config.model == "best":
        if logger: logger.info("Using best model")
        # TODO
        raise NotImplementedError("Best model not implemented yet")
    else:
        raise ValueError(f"Model {config.model} not implemented! Choose from: "
                         f"dummy, baseline, gru, lstm, xlstm, mamba, transformer, best")
