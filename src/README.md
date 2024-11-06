# Source Code of Masters Thesis 

## Execute
First either 
- configure existing config files (check `config` folder) and run as described below
- create a custom config yml file and provide it as a flag to the main script
  - to run custom models, you first need to create a new models folder and update the respective parts in `main.py` and the `get_model` function in [get_model.py](models/get_model.py)

From within src
```shell
python main.py
```

With the flags of the respective models (below), can run model training:
```shell
python main.py -f
```

With the possible flags:
- `-c` or `--custom_path` to specify a custom path to the config file
- `-d` or `--dummy` to use the dummy config (for debugging/testing)
- `-b` or `--baseline` to use the baseline config (simple MLP)
- `-l` or `--lstm` to use the LSTM config
- `-g` or `--gru` to use the GRU config
- `-x` or `--xlstm` to use the xLSTM config
- `-m` or `--mamba` to use the Mamba config
- `-j` or `--jamba` to use the Jamba config (TODO not working, cuda OOM)
- `-t` or `--transformer` to use the transformer config


## TODO
- remove `torchxlstm` model if not needed
