from __future__ import division
from __future__ import print_function

from models.pytorch.gat.layer import GATLayer
from multitask_benchmark.util.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
parser.add_argument('--nheads', type=int, default=4, help='Number of attentions heads.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
args = parser.parse_args()

execute_train(gnn_args=dict(nfeat=None,
                            nhid=args.hidden,
                            nodes_out=None,
                            graph_out=None,
                            dropout=args.dropout,
                            device=None,
                            first_conv_descr=dict(layer_type=GATLayer,
                                                  args=dict(
                                                      nheads=args.nheads,
                                                      alpha=args.alpha
                                                  )),
                            middle_conv_descr=dict(layer_type=GATLayer,
                                                   args=dict(
                                                       nheads=args.nheads,
                                                       alpha=args.alpha
                                                   )),
                            fc_layers=args.fc_layers,
                            conv_layers=args.conv_layers,
                            skip=args.skip,
                            gru=args.gru,
                            fixed=args.fixed,
                            variable=args.variable), args=args)
