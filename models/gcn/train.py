from __future__ import division
from __future__ import print_function

from models.gnn_framework import GNN
from models.gcn.layer import GCNLayer
from util.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
args = parser.parse_args()

execute_train(gnn_args=dict(nfeat=None,
                            nhid=args.hidden,
                            nodes_out=None,
                            graph_out=None,
                            dropout=args.dropout,
                            device=None,
                            first_conv_descr=dict(layer_type=GCNLayer, args=dict()),
                            middle_conv_descr=dict(layer_type=GCNLayer, args=dict()),
                            fc_layers=args.fc_layers,
                            conv_layers=args.conv_layers,
                            skip=args.skip,
                            gru=args.gru,
                            fixed=args.fixed,
                            variable=args.variable), args=args)
