import torch
from torch import Tensor
from typing import Dict

# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['lin'] / deg
    scale[deg == 0] = 1
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}
