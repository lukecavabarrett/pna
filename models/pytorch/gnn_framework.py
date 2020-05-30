import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GRU, S2SReadout, MLP


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nodes_out, graph_out, dropout, conv_layers=2, fc_layers=3, first_conv_descr=None,
                 middle_conv_descr=None, final_activation='LeakyReLU', skip=False, gru=False, fixed=False,
                 variable=False, device='cpu'):
        """
        :param nfeat:               number of input features per node
        :param nhid:                number of hidden features per node
        :param nodes_out:           number of nodes' labels
        :param graph_out:           number of graph labels
        :param dropout:             dropout value
        :param conv_layers:         if variable, conv_layers should be a function : adj -> int, otherwise an int
        :param fc_layers:           number of fully connected layers before the labels
        :param first_conv_descr:    dict or SimpleNamespace: "type"-> type of layer, "args" -> dict of calling args
        :param middle_conv_descr:   dict or SimpleNamespace : "type"-> type of layer, "args" -> dict of calling args
        :param final_activation:    activation to be used on the last fc layer before the labels
        :param skip:                whether to use skip connections feeding to the readout
        :param gru:                 whether to use a shared GRU after each convolution
        :param fixed:               whether to reuse the same middle convolutional layer multiple times
        :param variable:            whether the number of convolutional layers is variable or fixed
        :param device:              device used for computation
        """
        super(GNN, self).__init__()
        if variable:
            assert callable(conv_layers), "conv_layers should be a function from adjacency matrix to int"
            assert fixed, "With a variable number of layers they must be fixed"
            assert not skip, "cannot have skip and fixed at the same time"
        else:
            assert type(conv_layers) == int, "conv_layers should be an int"
            assert conv_layers > 0, "conv_layers should be greater than 0"

        if type(first_conv_descr) == dict:
            first_conv_descr = types.SimpleNamespace(**first_conv_descr)
        assert type(first_conv_descr) == types.SimpleNamespace, "first_conv_descr should be dict or SimpleNamespace"
        if type(first_conv_descr.args) == dict:
            first_conv_descr.args = types.SimpleNamespace(**first_conv_descr.args)
        assert type(first_conv_descr.args) == types.SimpleNamespace, \
            "first_conv_descr.args should be either a dict or a SimpleNamespace"

        if type(middle_conv_descr) == dict:
            middle_conv_descr = types.SimpleNamespace(**middle_conv_descr)
        assert type(middle_conv_descr) == types.SimpleNamespace, "middle_conv_descr should be dict or SimpleNamespace"
        if type(middle_conv_descr.args) == dict:
            middle_conv_descr.args = types.SimpleNamespace(**middle_conv_descr.args)
        assert type(middle_conv_descr.args) == types.SimpleNamespace, \
            "middle_conv_descr.args should be either a dict or a SimpleNamespace"

        self.dropout = dropout
        self.conv_layers = nn.ModuleList()
        self.skip = skip
        self.fixed = fixed
        self.variable = variable
        self.n_fixed_conv = conv_layers
        self.gru = GRU(input_size=nhid, hidden_size=nhid, device=device) if gru else None

        # first graph convolution
        first_conv_descr.args.in_features = nfeat
        first_conv_descr.args.out_features = nhid
        first_conv_descr.args.device = device
        self.conv_layers.append(first_conv_descr.layer_type(**vars(first_conv_descr.args)))

        # middle graph convolutions
        middle_conv_descr.args.in_features = nhid
        middle_conv_descr.args.out_features = nhid
        middle_conv_descr.args.device = device
        for l in range(1 if fixed else conv_layers - 1):
            self.conv_layers.append(
                middle_conv_descr.layer_type(**vars(middle_conv_descr.args)))

        n_conv_out = nfeat + conv_layers * nhid if skip else nhid

        # nodes output: fully connected layers
        self.nodes_read_out = MLP(in_size=n_conv_out, hidden_size=n_conv_out, out_size=nodes_out, layers=fc_layers,
                                  mid_activation="LeakyReLU", last_activation=final_activation, device=device)

        # graph output: S2S readout
        self.graph_read_out = S2SReadout(n_conv_out, n_conv_out, graph_out, fc_layers=fc_layers, device=device,
                                         final_activation=final_activation)

    def forward(self, x, adj):
        # graph convolutions
        skip_connections = [x] if self.skip else None

        n_layers = self.n_fixed_conv(adj) if self.variable else self.n_fixed_conv
        conv_layers = [self.conv_layers[0]] + ([self.conv_layers[1]] * (n_layers - 1)) if self.fixed else self.conv_layers

        for layer, conv in enumerate(conv_layers):
            y = conv(x, adj)
            x = y if self.gru is None else self.gru(x, y)

            if self.skip:
                skip_connections.append(x)

            # dropout at all layers but the last
            if layer != n_layers - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        if self.skip:
            x = torch.cat(skip_connections, dim=2)

        # readout output
        return (self.nodes_read_out(x), self.graph_read_out(x))
