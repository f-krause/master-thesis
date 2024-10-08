# Source Code of Masters Thesis 

## Execute
First configure config file and specify in ``main.py`` 

From within src
```shell
python main.py
```

With the flags of the respective models, see:
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
