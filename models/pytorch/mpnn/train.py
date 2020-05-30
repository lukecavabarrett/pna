from __future__ import division
from __future__ import print_function

from models.pytorch.pna.layer import PNALayer
from multitask_benchmark.util.train import execute_train, build_arg_parser

# Training settings
parser = build_arg_parser()
parser.add_argument('--self_loop', action='store_true', default=False, help='Whether to add self loops in aggregators')
parser.add_argument('--towers', type=int, default=4, help='Number of towers in MPNN layers')
parser.add_argument('--aggregation', type=str, default='sum', help='Type of aggregation')
parser.add_argument('--pretrans_layers', type=int, default=1, help='Number of MLP layers before aggregation')
parser.add_argument('--posttrans_layers', type=int, default=1, help='Number of MLP layers after aggregation')
args = parser.parse_args()

# The MPNNs can be considered a particular case of PNA networks with a single aggregator and no scalers (identity)

execute_train(gnn_args=dict(nfeat=None,
                            nhid=args.hidden,
                            nodes_out=None,
                            graph_out=None,
                            dropout=args.dropout,
                            device=None,
                            first_conv_descr=dict(layer_type=PNALayer,
                                                  args=dict(
                                                      aggregators=[args.aggregation],
                                                      scalers=['identity'], avg_d=None,
                                                      towers=args.towers,
                                                      self_loop=args.self_loop,
                                                      divide_input=False,
                                                      pretrans_layers=args.pretrans_layers,
                                                      posttrans_layers=args.posttrans_layers
                                                  )),
                            middle_conv_descr=dict(layer_type=PNALayer,
                                                   args=dict(
                                                       aggregators=[args.aggregation],
                                                       scalers=['identity'],
                                                       avg_d=None, towers=args.towers,
                                                       self_loop=args.self_loop,
                                                       divide_input=True,
                                                       pretrans_layers=args.pretrans_layers,
                                                       posttrans_layers=args.posttrans_layers
                                                   )),
                            fc_layers=args.fc_layers,
                            conv_layers=args.conv_layers,
                            skip=args.skip,
                            gru=args.gru,
                            fixed=args.fixed,
                            variable=args.variable), args=args)
