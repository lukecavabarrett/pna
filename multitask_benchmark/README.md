# Multi-task benchmark

## Overview

We provide the scripts for the generation and execution of the multi-task benchmark.
- `dataset_generation/` contains:
  - the implementation of the aggregators, the scalers and the PNA layer (`/pna/`)
  - the flexible GNN framework that can be used with any type of graph convolutions (`gnn_framework.py`)
  - implementations of the other GNN models used for comparison in the paper, namely GCN, GAT, GIN and MPNN
- `util/` contains
  - preprocessing subroutines and loss functions (`util.py`)
  - training and evaluation procedures (`train.py`) 
  
This benchmark uses the PyTorch version of PNA (`../models/pytorch/pna/`)

## Dependencies
Install PyTorch from the [official website](https://pytorch.org/). The code was tested over PyTorch 1.4.

Then install the other dependencies:
```
pip3 install -r multitask_benchmark/requirements.txt
```

## Test run

Move to the source of the repository before running the following.

Generate the benchmark dataset (add `--extrapolation` for multiple test sets of different sizes):
```
python3 -m multitask_benchmark.datasets_generation.multitask_dataset
```

then run the training, for example:
```
python3 -m multitask_benchmark.train.pna --variable --fixed --gru --variable_conv_layers=N/2 --aggregators="mean max min std" --scalers="identity amplification attenuation" --data=multitask_benchmark/data/multitask_dataset.pkl
```
