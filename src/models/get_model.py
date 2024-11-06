import torch
from omegaconf import DictConfig


def get_model(config: DictConfig, device: torch.device, logger=None):
    if config.frequency_features and config.model not in ["baseline", "transformer"]:
        raise ValueError(f"Frequency features only supported for baseline and transformer models, not {config.model}")

    if config.model == "dummy":
        if logger: logger.warning("Using dummy model")
        from models.dummy.dummy import ModelDummy
        return ModelDummy(device).to(device)
    elif config.model == "baseline":
        if config.frequency_features:
            if logger: logger.info("Using baseline model (frequencies)")
            from models.baseline.baseline_freq import ModelBaseline
        else:
            if logger: logger.info("Using baseline model")
            from models.baseline.baseline import ModelBaseline
        return ModelBaseline(config, device).to(device)
    elif config.model == "cnn":
        from models.cnn.cnn import ModelCNN
        return ModelCNN(config, device).to(device)
    elif config.model == "gru":
        if logger: logger.info("Using GRU model")
        from models.rnn.rnn import ModelRNN
        return ModelRNN(config, device, model="gru").to(device)
    elif config.model == "lstm":
        if logger: logger.info("Using LSTM model")
        from models.rnn.rnn import ModelRNN
        return ModelRNN(config, device, model="lstm").to(device)
    elif config.model == "xlstm":
        if logger: logger.info("Using xLSTM model")
        from models.xlstm.xlstm import ModelXLSTM
        return ModelXLSTM(config, device).to(device)
    elif config.model == "mamba":
        if logger: logger.info("Using Mamba model")
        from models.mamba.mamba import ModelMamba
        return ModelMamba(config, device, model="mamba").to(device)
    elif config.model == "mamba2":
        if logger: logger.info("Using Mamba2 model")
        from models.mamba.mamba import ModelMamba
        return ModelMamba(config, device, model="mamba2").to(device)
    elif config.model == "jamba":
        if logger: logger.info("Using Jamba model")
        from models.jamba.jamba import ModelJamba
        return ModelJamba(config, device).to(device)
    elif config.model == "transformer":
        if config.frequency_features:
            if logger: logger.info("Using Transformer model (frequencies)")
            from models.transformer.transformer_freq import ModelTransformer  # codon frequencies
        else:
            if logger: logger.info("Using Transformer model")
            from models.transformer.transformer import ModelTransformer
        return ModelTransformer(config, device).to(device)
    elif config.model.lower() == "tisnet":
        if logger: logger.info("Using TISnet model")
        from models.TISnet.TISnet import TISnet
        return TISnet(config, device).to(device)
    elif config.model == "best":
        if logger: logger.info("Using best model")
        # TODO
        raise NotImplementedError("Best model not implemented yet")
    else:
        raise ValueError(f"Model {config.model} not implemented! Choose from: "
                         f"dummy, baseline, cnn, gru, lstm, xlstm, mamba, transformer, best")  # TODO update
