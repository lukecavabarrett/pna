import torch
from torch_scatter import scatter_sum, scatter_mean, scatter_max, scatter_min

EPS = 1e-5


def aggregate_sum(src, index, dim, dim_size):
    return scatter_sum(src=src, index=index, dim=dim, out=None, dim_size=dim_size)


def aggregate_mean(src, index, dim, dim_size):
    return scatter_mean(src=src, index=index, dim=dim, out=None, dim_size=dim_size)


def aggregate_max(src, index, dim, dim_size):
    return scatter_max(src=src, index=index, dim=dim, out=None, dim_size=dim_size)[0]


def aggregate_min(src, index, dim, dim_size):
    return scatter_min(src=src, index=index, dim=dim, out=None, dim_size=dim_size)[0]


def aggregate_var(src, index, dim, dim_size):
    mean = aggregate_mean(src, index, dim, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim, dim_size)
    var = mean_squares - mean * mean
    return var


def aggregate_std(src, index, dim, dim_size):
    var = aggregate_var(src, index, dim, dim_size)
    out = torch.sqrt(torch.relu(var) + EPS)
    return out


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var}