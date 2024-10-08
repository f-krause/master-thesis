# Master Thesis Repo

Main Repository for Data Science Master Thesis at University of Vienna 2024

## Link selection

[command collection](https://git01lab.cs.univie.ac.at/a1142469/dap/-/blob/main/RNAdegformer/command_collection.md?ref_type=heads) 
used in DAP 2024 on university servers


## Set up environment
Create virtual environment with mamba/conda
```shell
mamba env create -f environment_files/environment_linux_py3.10.yml
mamba activate master-env
```

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




## Command collection
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


### Environment
Export environment dependencies
```shell
mamba env export -n master-env > environment_files/environment_linux_py3.10.yml
```

Update environment dependencies
```shell
mamba env update -n master-env -f environment_files/environment_linux_py3.10.yml
```

### use AIM logging
```shell
aim up
```

```shell
ssh -f -N -L 43800:localhost:43800 krausef99dm@jyn.dm.univie.ac.at
```


### Other 
Count files in a directory
```shell
ls -1 | wc -l
```


## More
TBW


