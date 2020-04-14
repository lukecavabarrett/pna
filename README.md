# Principal Neighbourhood Aggregation

Principal Neighbourhood Aggregation for GNNs [https://arxiv.org/abs/2004.05718](https://arxiv.org/abs/2004.05718)

## Overview

We provide the implementation of the Principal Neighbourhood Aggregation (PNA) in PyTorch along with scripts to generate the multitask benchmarks, a flexible GNN framework and implementations of the other models used for comparison. The repository is organised as follows:
- `datasets_generation/` contains the scripts for the generation of the benchmarks
- `models/` contains:
  - the implementation of the aggregators, the scalers and the PNA layer (`/pna/`)
  - the flexible GNN framework that can be used with any type of graph convolutions (`gnn_framework.py`)
  - implementations of the other GNN models used for comparison in the paper, namely GCN, GAT, GIN and MPNN
- `util/` contains
  - preprocessing subroutines and loss functions (`util.py`)
  - training and evaluation procedures (`train.py`)
  - general NN layers used by the various models (`layers.py`) 

## Dependencies
Install PyTorch from the [official website](https://pytorch.org/). The code was tested over PyTorch 1.4.

Then install the other dependencies:
```
pip install -r requirements.txt
```

## Test run

Generate the benchmark dataset (add `--extrapolation` for multiple test sets of different sizes):
```
python -m datasets_generation.multitask_dataset
```

then run the training, for example:
```
python -m models.pna.train --variable --fixed --gru --variable_conv_layers=N/2 --aggregators="mean max min std" --scalers="identity amplification attenuation" --data=data/multitask_dataset.pkl
```

The model specified by the arguments above uses the same architecture (represented in the image below) and the same aggregators and scalers as used for the results in the paper. Note that the default hyper-parameters are not the best performing for every model, refer to the paper for details on how we set them.

![Architecture](PNA_architecture.png)


## Reference
```
@misc{corso2020principal,
    title={Principal Neighbourhood Aggregation for Graph Nets},
    author={Gabriele Corso and Luca Cavalleri and Dominique Beaini and Pietro Liò and Petar Veličković},
    year={2020},
    eprint={2004.05718},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License
MIT
