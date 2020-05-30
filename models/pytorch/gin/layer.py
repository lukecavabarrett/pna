import torch
import torch.nn as nn
from models.layers import MLP


class GINLayer(nn.Module):
    """
        Graph Isomorphism Network layer, similar to https://arxiv.org/abs/1810.00826
    """

    def __init__(self, in_features, out_features, fc_layers=2, device='cpu'):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param fc_layers:       number of fully connected layers after the sum aggregator
        :param device:          device used for computation
        """
        super(GINLayer, self).__init__()

        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = nn.Parameter(torch.zeros(size=(1,), device=device))
        self.post_transformation = MLP(in_size=in_features, hidden_size=max(in_features, out_features),
                                       out_size=out_features, layers=fc_layers, mid_activation='relu',
                                       last_activation='relu', mid_b_norm=True, last_b_norm=False, device=device)
        self.reset_parameters()

    def reset_parameters(self):
        self.epsilon.data.fill_(0.1)

    def forward(self, input, adj):
        (B, N, _) = adj.shape

        # sum aggregation
        mod_adj = adj + torch.eye(N, device=self.device).unsqueeze(0) * (1 + self.epsilon)
        support = torch.matmul(mod_adj, input)

        # post-aggregation transformation
        return self.post_transformation(support)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
