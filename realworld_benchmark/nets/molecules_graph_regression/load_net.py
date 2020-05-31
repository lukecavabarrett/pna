"""
    Utility file to select GraphNN model as
    selected by the user
"""

from .pna_net import PNANet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def MoNet(net_params):
    return MoNet_(net_params)


def DiffPool(net_params):
    return DiffPoolNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)


def PNA(net_params):
    return PNANet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'DiffPool': DiffPool,
        'MLP': MLP,
        'PNA': PNA

    }

    return models[MODEL_NAME](net_params)
