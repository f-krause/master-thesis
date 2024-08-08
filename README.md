# Master Thesis Repo

Main Repository for Data Science Master Thesis at University of Vienna 2024

## Link selection

[command collection](https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/command_collection.md?ref_type=heads) 
used in DAP 2024 on university servers


## Set up environment
Create virtual environment with mamba/conda
```shell
mamba env create -f environment.yml
mamba activate master-env
```

Create project structure, and specify project path and subproject in ``main.py``
```
root
├── krausef99dm_thesis  (project folder)
│   ├── data
│   ├── dev
|   │   ├── logs
|   │   ├── weights
|   │   ├── weights_best
│   ├── lstm
|   │   ├── ...
│   ├── xlstm
|   │   ├── ...
│   ├── gru
|   │   ├── ...
│   ├── transformer
|   │   ├── ...
└── master-thesis  (this repo)
```

Install BPP prediction folding algorithms. Follow [arnie tutorial](https://github.com/DasLab/arnie/blob/master/docs/setup_doc.md).




## Command collection
Connect to the university server with the following command:
```shell
ssh krausef99dm@rey.dm.univie.ac.at
```

Export environment dependencies
```shell
mamba env export -n master-env > environment.yml
```

Update environment dependencies
```shell
mamba env update -n master-env -f environment.yml
```


## More
TBW


