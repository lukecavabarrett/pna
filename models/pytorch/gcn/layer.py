import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
        GCN layer, similar to https://arxiv.org/abs/1609.02907
        Implementation inspired by https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param bias:            whether to add a learnable bias before the activation
        :param device:          device used for computation
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), device=device))
        if bias:
            self.b = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, X, adj):
        (B, N, _) = adj.shape

        # linear transformation
        XW = torch.matmul(X, self.W)

        # normalised mean aggregation
        adj = adj + torch.eye(N, device=self.device).unsqueeze(0)
        rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5),
                       torch.eye(N, device=self.device).unsqueeze(0))  # D^{-1/2]
        adj = torch.matmul(torch.matmul(rD, adj), rD)  # D^{-1/2] A' D^{-1/2]
        y = torch.bmm(adj, XW)

        if self.b is not None:
            y = y + self.b
        return F.leaky_relu(y)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
