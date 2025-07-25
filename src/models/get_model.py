import torch
from omegaconf import DictConfig


def get_model(config: DictConfig, device: torch.device, logger=None):
    if config.frequency_features and config.model.lower() not in ["baseline", "transformer", "ptrnet"]:
        raise ValueError(f"Frequency features only supported for baseline and transformer models, not {config.model}")
    elif config.model.lower() == "baseline":
        if config.frequency_features:
            if logger: logger.info("Using baseline model (frequencies)")
            from models.baseline.baseline_freq import ModelBaseline
        else:
            if logger: logger.info("Using baseline model")
            from models.baseline.baseline import ModelBaseline
        return ModelBaseline(config, device).to(device)
    elif config.model.lower() == "cnn":
        from models.cnn.cnn import ModelCNN
        return ModelCNN(config, device).to(device)
    elif config.model.lower() == "gru":
        if logger: logger.info("Using GRU model")
        from models.rnn.rnn import ModelRNN
        return ModelRNN(config, device, model="gru").to(device)
        # from models.rnn.rnn_mean_regression import ModelGRUMeanRegression
        # return ModelGRUMeanRegression(config, device).to(device)
    elif config.model.lower() == "lstm":
        if logger: logger.info("Using LSTM model")
        from models.rnn.rnn import ModelRNN
        return ModelRNN(config, device, model="lstm").to(device)
    elif config.model.lower() == "xlstm":
        if logger: logger.info("Using xLSTM model")
        from models.xlstm.xlstm import ModelXLSTM
        return ModelXLSTM(config, device).to(device)
    elif config.model.lower() == "mamba":
        if logger: logger.info("Using Mamba model")
        from models.mamba.mamba import ModelMamba
        return ModelMamba(config, device, model="mamba").to(device)
    elif config.model.lower() == "mamba2":
        if logger: logger.info("Using Mamba2 model")
        from models.mamba.mamba import ModelMamba
        return ModelMamba(config, device, model="mamba2").to(device)
    elif config.model.lower() == "transformer":
        if config.frequency_features:
            if logger: logger.info("Using Transformer model (frequencies)")
            from models.transformer.transformer_freq import ModelTransformer  # codon frequencies
        else:
            if logger: logger.info("Using Transformer model")
            from models.transformer.transformer import ModelTransformer
        return ModelTransformer(config, device).to(device)
    elif config.model.lower() == "legnet":
        if logger: logger.info("Using LegNet model")
        from models.LegNet.LegNet import LegNet
        return LegNet(config, device).to(device)
    elif config.model.lower() == "ribonn":
        if logger: logger.info("Using RiboNN model")
        from models.RiboNN.RiboNN import RiboNN
        return RiboNN(config, device).to(device)
    elif config.model.lower() == "ptrnet":
        if logger: logger.info("Using PTRnet model")
        from models.PTRnet.PTRnet import PTRnet
        return PTRnet(config, device).to(device)
    else:
        raise ValueError(f"Model {config.model} not implemented! Choose from: "
                         f"baseline, cnn, gru, lstm, xlstm, mamba, transformer, LegNet, RiboNN, best")  # TODO update
