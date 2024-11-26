# Source Code of Masters Thesis 

## Run model training
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


## Optuna Hyperparameter Optimization
1. Set general study parameters in [`general_codon.yml`](src/config/general_codon.yml) config
```yaml
optuna:
  storage: 'sqlite:////export/share/krausef99dm/tuning_dbs/'  # only provide path, will add database per model
  study_name: 'study_1'
  n_trials: 5
  timeout: 86400  # for 24h max runtime
```

2. Set parameters to try in the respective yaml files in [`config/param_tuning`](config/param_tuning)

3. Run hyperparameter optimization with Optuna
```shell
python main.py --tuning  --model_name
```

4. Get dashboard for Optuna study
```shell
optuna-dashboard sqlite:////export/share/krausef99dm/tuning_dbs/dev/gru.db
```
