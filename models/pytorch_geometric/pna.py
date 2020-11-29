from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree

from models.pytorch_geometric.aggregators import AGGREGATORS
from models.pytorch_geometric.scalers import SCALERS

# Implemented with the help of Matthias Fey, author of PyTorch Geometric
# For an example see https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pna.py

class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper
        .. math::
            \bigoplus = \underbrace{\begin{bmatrix}I \\ S(D, \alpha=1) \\
            S(D, \alpha=-1) \end{bmatrix} }_{\text{scalers}}
            \otimes \underbrace{\begin{bmatrix} \mu \\ \sigma \\ \max \\ \min
            \end{bmatrix}}_{\text{aggregators}},
        in:
        .. math::
            X_i^{(t+1)} = U \left( X_i^{(t)}, \underset{(j,i) \in E}{\bigoplus}
            M \left( X_i^{(t)}, X_j^{(t)} \right) \right)
        where :math:`M` and :math:`U` denote the MLP referred to with pretrans
        and posttrans respectively.
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggregators (list of str): Set of aggregation function identifiers,
                namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
                :obj:`"var"` and :obj:`"std"`.
            scalers: (list of str): Set of scaling function identifiers, namely
                :obj:`"identity"`, :obj:`"amplification"`,
                :obj:`"attenuation"`, :obj:`"linear"` and
                :obj:`"inverse_linear"`.
            deg (Tensor): Histogram of in-degrees of nodes in the training set,
                used by scalers to normalize.
            edge_dim (int, optional): Edge feature dimensionality (in case
                there are any). (default :obj:`None`)
            towers (int, optional): Number of towers (default: :obj:`1`).
            pre_layers (int, optional): Number of transformation layers before
                aggregation (default: :obj:`1`).
            post_layers (int, optional): Number of transformation layers after
                aggregation (default: :obj:`1`).
            divide_input (bool, optional): Whether the input features should
                be split between towers or not (default: :obj:`False`).
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        super(PNAConv, self).__init__(aggr=None, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, dim={self.dim})')
        raise NotImplementedError


class PNAConvSimple(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper
        .. math::
            \bigoplus = \underbrace{\begin{bmatrix}I \\ S(D, \alpha=1) \\
            S(D, \alpha=-1) \end{bmatrix} }_{\text{scalers}}
            \otimes \underbrace{\begin{bmatrix} \mu \\ \sigma \\ \max \\ \min
            \end{bmatrix}}_{\text{aggregators}},
        in:
        .. math::
            X_i^{(t+1)} = U \left( \underset{(j,i) \in E}{\bigoplus}
            M \left(X_j^{(t)} \right) \right)
        where :math:`U` denote the MLP referred to with posttrans.
        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            aggregators (list of str): Set of aggregation function identifiers,
                namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
                :obj:`"var"` and :obj:`"std"`.
            scalers: (list of str): Set of scaling function identifiers, namely
                :obj:`"identity"`, :obj:`"amplification"`,
                :obj:`"attenuation"`, :obj:`"linear"` and
                :obj:`"inverse_linear"`.
            deg (Tensor): Histogram of in-degrees of nodes in the training set,
                used by scalers to normalize.
            post_layers (int, optional): Number of transformation layers after
                aggregation (default: :obj:`1`).
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 post_layers: int = 1, **kwargs):

        super(PNAConvSimple, self).__init__(aggr=None, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        self.F_in = in_channels
        self.F_out = self.out_channels

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        in_channels = (len(aggregators) * len(scalers)) * self.F_in
        modules = [Linear(in_channels, self.F_out)]
        for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_out, self.F_out)]
        self.post_nn = Sequential(*modules)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.post_nn)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)
        return self.post_nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')
        raise NotImplementedError