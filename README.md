# Master Thesis Repo

Main Repository for Data Science Master Thesis at University of Vienna 2024

## Link selection

- [command collection](https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/command_collection.md?ref_type=heads) used in DAP 2024 on university servers
- [GENCODE glossary](https://www.gencodegenes.org/pages/data_format.html) - Format description
- [PTR ratios source](https://figshare.com/articles/dataset/Additional_file_2_Protein-to-mRNA_ratios_among_tissues/21379197?file=37938894)

## Set up environment
Create virtual environment with mamba/conda
```shell
mamba env create -f environment_files/environment_linux_py3.10.yml
mamba activate master-env
```

Set up Ranger Optimizer
```shell
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e . 
```

## Data Structure (UPDATE)
Create project structure, and specify project path and subproject in ``main.py``
```
root
├── krausef99dm  (project folder)
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

Install BPP prediction folding algorithms. Follow [arnie tutorial](https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md).


## File Structure
### 📦 Repository Overview

This repository supports a deep learning benchmark study for predicting protein-to-mRNA (PTR) ratios using mRNA sequence and structure features.

#### 🔧 Configuration
- `config/`: YAML configs for model architecture, training, and hyperparameter tuning (e.g., for Mamba, LSTM, Transformer, etc.).

#### 📊 Data Handling
- `data_handling/`: Scripts for preprocessing, structure prediction, codon/nucleotide dataset creation, and stratified splitting.

#### 🧠 Models
- `models/`: Implementation of deep learning models (MLP, CNN, RNNs, Transformer, xLSTM, Mamba, LegNet, PTRnet).
- Modularized by model type with shared predictor logic.

#### 🎓 Pretraining
- `pretraining/`: Tools for masked language model (MLM) pretraining and motif discovery.

#### 🏋️ Training
- `training/`: Training logic, early stopping, learning rate scheduling, and Optuna-based tuning.

#### 📈 Evaluation
- `evaluation/`: Model evaluation, metrics, predictions, and plotting utilities.

#### 🛠️ Utilities
- `utils/`, `log/`: Helper functions, logging setup, and device management.

#### 🚀 Entry Point
- `main.py`: Main script for running training or tuning, configurable via CLI flags.
- `multi_run*.sh`: Example scripts to train multiple models sequentially.


## Command Collection
### SSH connection
Connect to the university server with the following command:
```shell
ssh krausef99dm@rey.dm.univie.ac.at
```

```shell
ssh krausef99dm@jyn.dm.univie.ac.at
```

Data path
```shell
/export/share/krausef99dm/
```

Download a file from remote
```shell
scp krausef99dm@rey.dm.univie.ac.at:/export/share/krausef99dm/data/FILE C:/Users/Felix/code/uni/UniVie/master-thesis/data/FILE
```

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

```shell
ssh -f -N -L 43800:localhost:43800 krausef99dm@jyn.dm.univie.ac.at
```

How to search and filter runs in AIM: https://aimstack.readthedocs.io/en/latest/using/search.html



### Use Optuna dashboard
From within the data folder for optuna, run:
```shell
optuna-dashboard sqlite:////export/share/krausef99dm/tuning_dbs/baseline.db
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
