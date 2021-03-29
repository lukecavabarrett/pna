# Real-world benchmarks

<img src="https://raw.githubusercontent.com/lukecavabarrett/pna/master/multitask_benchmark/images/realworld_results.png" alt="Real world results" width="500"/>

## Overview

We provide the scripts for the download and execution of the real-world benchmarks we used. 
Many scripts in this directory were taken directly from or inspired by "Benchmarking GNNs" 
by Dwivedi _et al._ refer to their [code](https://github.com/graphdeeplearning/benchmarking-gnns) 
and [paper](https://arxiv.org/abs/2003.00982) for more details on their work. The graph classification
benchmark MolHIV comes from the [Open Graph Benchmark](https://ogb.stanford.edu/).

- `configs` contains .json configuration files for the various datasets;
- `data` contains scripts to download the datasets;
- `nets` contains the architectures that were used with the PNA in the benchmarks;
- `train` contains the training scripts.
  
These benchmarks use the DGL version of PNA (`../models/dgl`) with the MolHIV model using the *simple* layer architecture. 
Below you can find the instructions on how to download the datasets and run the models. 
You can run these scripts directly in this [notebook](https://colab.research.google.com/drive/1RnV4MBjCl98eubAGpEF-eXdAW5mTP3h3?usp=sharing).



## Test run

### Benchmark Setup

[Follow these instructions](./docs/setup.md) to install the benchmark and setup the environment.

### Run model training
```
# at the root of the repo
cd realworld_benchmark
python { main_molecules.py | main_superpixels.py } [--param=value ...] --dataset { ZINC | MNIST | CIFAR10 } --gpu_id gpu_id --config config_file
```


## Tuned hyperparameters

You can find below the hyperparameters we used for our experiments. In general, the depth of the architectures was not changed while the width was adjusted to keep the total number of parameters of the model between 100k and 110k as done in "Benchmarking GNNs" to ensure a fair comparison of the architectures. Refer to our [paper](https://arxiv.org/abs/2004.05718) for an interpretation of the results.

```
For the leaderboard (2nd version of the datasets - 400/500k parameters)

# ZINC
PNA:
python main_molecules.py --weight_decay=3e-6 --L=16 --hidden_dim=70 --out_dim=70 --residual=True --edge_feat=True --edge_dim=40 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20
MPNN (sum/max):
python main_molecules.py --weight_decay=3e-6 --L=16 --hidden_dim=110 --out_dim=110 --residual=True --edge_feat=True --edge_dim=40 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20


For the paper (1st version of the datasets - 100k parameters)
--- PNA ---

# ZINC
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=5
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=70 --out_dim=60 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20

# CIFAR10
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5

# MNIST
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=75 --out_dim=70 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5


--- PNA (no scalers) ---

# ZINC
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=95 --out_dim=90 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=5
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=90 --out_dim=80 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --pretrans_layers=1 --posttrans_layers=1 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20

# CIFAR10
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=95 --out_dim=90 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=95 --out_dim=90 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5

# MNIST
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=95 --out_dim=90 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.1 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=95 --out_dim=90 --residual=True --edge_feat=True --edge_dim=50 --readout=sum --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=True --batch_norm=True --aggregators="mean max min std" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5


--- MPNN (sum/max) ---

# ZINC
python main_molecules.py --weight_decay=1e-5 --L=4 --hidden_dim=110 --out_dim=80 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=5
python main_molecules.py --weight_decay=3e-6 --L=4 --hidden_dim=100 --out_dim=70 --residual=True --edge_dim=50 --edge_feat=True --readout=sum --in_feat_dropout=0.0 --dropout=0.0 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=False --divide_input_last=True  --dataset ZINC --gpu_id 0 --config "configs/molecules_graph_regression_pna_ZINC.json" --lr_schedule_patience=20

# CIFAR10
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=110 --out_dim=90 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.2 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=110 --out_dim=90 --residual=True --edge_feat=True --edge_dim=20 --readout=sum --in_feat_dropout=0.0 --dropout=0.2 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset CIFAR10 --gpu_id 0 --config "configs/superpixels_graph_classification_pna_CIFAR10.json" --lr_schedule_patience=5

# MNIST
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=110 --out_dim=90 --residual=True --edge_feat=False --readout=sum --in_feat_dropout=0.0 --dropout=0.2 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5
python main_superpixels.py --weight_decay=3e-6 --L=4 --hidden_dim=110 --out_dim=90 --residual=True --edge_feat=True --edge_dim=20 --readout=sum --in_feat_dropout=0.0 --dropout=0.2 --graph_norm=True --batch_norm=True --aggregators="sum"/"max" --scalers="identity" --towers=5 --divide_input_first=True --divide_input_last=False  --dataset MNIST --gpu_id 0 --config "configs/superpixels_graph_classification_pna_MNIST.json" --lr_schedule_patience=5

```

For OGB leaderboard run the following scripts in the [DGN](https://github.com/Saro00/DGN) repository:

```
# MolHIV 

python -m main_HIV --weight_decay=3e-6 --L=4 --hidden_dim=80 --out_dim=80 --residual=True --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --batch_norm=True --aggregators="mean max min std" --scalers="identity amplification attenuation" --dataset HIV --gpu_id 0 --config "configs/molecules_graph_classification_PNA_HIV.json" --epochs=200 --init_lr=0.01 --lr_reduce_factor=0.5 --lr_schedule_patience=20 --min_lr=0.0001

# MolPCBA 

python main_PCBA.py --type_net="complex" --batch_size=512 --lap_norm="none" --weight_decay=3e-6 --L=4 --hidden_dim=510 --out_dim=510 --residual=True --edge_feat=True  --readout=sum --graph_norm=True --batch_norm=True --aggregators="mean sum max" --scalers="identity" --gpu_id 0 --config "configs/PCBA_graph_classification_EIG.json"  --lr_schedule_patience=4 --towers=5 --dropout=0.2 --init_lr=0.0005 --min_lr=0.00002 --edge_dim=16 --lr_reduce_factor=0.8
```


