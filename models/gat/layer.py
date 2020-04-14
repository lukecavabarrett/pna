import torch
import torch.nn as nn
import torch.nn.functional as F


class GATHead(nn.Module):

    def __init__(self, in_features, out_features, alpha, activation=True, device='cpu'):
        super(GATHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), device=device))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1), device=device))
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=0.1414)
        nn.init.xavier_uniform_(self.a.data, gain=0.1414)

    def forward(self, input, adj):

        h = torch.matmul(input, self.W)
        (B, N, _) = adj.shape
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=1)\
            .view(B, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, h)

        if self.activation:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(nn.Module):
    """
        Graph Attention Layer, GAT paper at https://arxiv.org/abs/1710.10903
        Implementation inspired by https://github.com/Diego999/pyGAT
    """

    def __init__(self, in_features, out_features, alpha, nheads=1, activation=True, device='cpu'):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param alpha:           slope of the leaky relu
        :param nheads:          number of attention heads
        :param activation:      whether to apply a non-linearity
        :param device:          device used for computation
        """
        super(GATLayer, self).__init__()
        assert (out_features % nheads == 0)

        self.input_head = in_features
        self.output_head = out_features // nheads

        self.heads = nn.ModuleList()
        for _ in range(nheads):
            self.heads.append(GATHead(in_features=self.input_head, out_features=self.output_head, alpha=alpha,
                                      activation=activation, device=device))

    def forward(self, input, adj):
        y = torch.cat([head(input, adj) for head in self.heads], dim=2)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
