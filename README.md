# Master Thesis Repo

Main Repository for Data Science Master Thesis at University of Vienna 2024-25:

"Benchmarking and Optimizing Deep Learning Architectures for Protein-to-mRNA Ratio Prediction"


## Link selection
- [Reference paper](https://link.springer.com/article/10.1186/s13059-023-02868-2): Hernandez-Alias et al (2023), Using protein-per-mRNA differences among human tissues in codon optimization
- [GENCODE data format glossary](https://www.gencodegenes.org/pages/data_format.html)
- [PTR ratios data source](https://figshare.com/articles/dataset/Additional_file_2_Protein-to-mRNA_ratios_among_tissues/21379197?file=37938894)


## Set up environment
Create virtual environment with mamba/conda
```shell
mamba env create -f environment_files/environment_linux_py3.10.yml
mamba activate master-env
```


## Data Structure
Create a project folder for training (`data`) and model data (`runs`), and specify project path in [``src/utils/utils.py``](src/utils/utils.py) in the function `set_project_path()`.
```
root
├── project folder
│   ├── data
│   ├── runs
│   │   ├── dev
│   │   │   ├── logs
│   │   │   ├── weights
│   │   │   ├── weights_best
│   │   ├── lstm
│   │   │   ├── ...
│   │   ├── xlstm
│   │   │   ├── ...
│   │   ├── gru
│   │   │   ├── ...
│   │   ├── transformer
│   │   │   ├── ...
│   │   ├── best_model (PTRNet)
│   │   │   ├── ...
└── master-thesis  (this repo)
```


## 📦 Repository Overview
This repository supports a deep learning benchmark study for predicting protein-to-mRNA (PTR) ratios using mRNA sequence and structure features.

#### 🔧 Configuration
- `src/config/`: YAML configs for model architecture, training, and hyperparameter tuning (e.g., for Mamba, LSTM, Transformer, etc.).

#### 📊 Data Handling
- `src/data_handling/`: Scripts for preprocessing, structure prediction, codon/nucleotide dataset creation, and stratified splitting.

#### 🧠 Models
- `src/models/`: Implementation of deep learning models (MLP, CNN, RNNs, Transformer, xLSTM, Mamba, LegNet, PTRnet).
- Modularized by model type with shared predictor logic.

#### 🎓 Pretraining
- `src/pretraining/`: Tools for masked language model (MLM) pretraining and motif discovery.

#### 🏋️ Training
- `src/training/`: Training logic, early stopping, learning rate scheduling, and Optuna-based tuning.

#### 📈 Evaluation
- `src/evaluation/`: Model evaluation, metrics, predictions, and plotting utilities.

#### 🛠️ Utilities
- `src/utils/`, `src/log/`: Helper functions, logging setup, and device management.

#### 🚀 Entry Point
- `src/main.py`: Main script for running training or tuning, configurable via CLI flags.
- `src/multi_run*.sh`: Example scripts to train multiple models sequentially.



## Set up structure prediction

Install folding algorithms for secondary structure predictions. Follow [arnie tutorial](https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md).

The [bpRNA code](https://github.com/hendrixlab/bpRNA/tree/master) for loop type predictions is already in the repo.



## Command collection
### Environment
Export environment dependencies
```shell
mamba env export -n master-env > environment_files/environment_linux_py3.10.yml
```

Update environment dependencies
```shell
mamba env update -n master-env -f environment_files/environment_linux_py3.10.yml
```

### Use AIM logging
```shell
aim up
```

On how to search and filter runs in AIM: https://aimstack.readthedocs.io/en/latest/using/search.html



### Use Optuna dashboard
From within the data folder for optuna, run:
```shell
optuna-dashboard sqlite:////path/to/optuna/model_name.db
```


### start jupyter server manually
```shell
jupyter notebook --no-browser --port=8888
```


### Other 
Count files in a directory
```shell
ls -1 | wc -l
```
