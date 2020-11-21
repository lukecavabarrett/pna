import torch.nn as nn
import dgl
from models.dgl.pna_layer import PNASimpleLayer
from nets.mlp_readout_layer import MLPReadout
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder


class PNANet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        posttrans_layers = net_params['posttrans_layers']
        device = net_params['device']
        self.device = device

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        self.layers = nn.ModuleList(
            [PNASimpleLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, posttrans_layers=posttrans_layers)
             for _ in range(n_layers - 1)])
        self.layers.append(PNASimpleLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, posttrans_layers=posttrans_layers))

        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc_dim > 0:
            h_pos_enc = self.embedding_pos_enc(g.ndata['pos_enc'].to(self.device))
            h = h + h_pos_enc
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            if self.gru_enable and i != len(self.layers) - 1:
                h_t = self.gru(h, h_t)
            h = h_t

        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        loss = torch.nn.BCEWithLogitsLoss()(scores, labels.type(torch.FloatTensor).to('cuda').unsqueeze(-1))
        return loss
