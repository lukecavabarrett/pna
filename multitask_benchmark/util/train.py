from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
from types import SimpleNamespace

import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from models.pytorch.gnn_framework import GNN
from multitask_benchmark.util.util import load_dataset, total_loss, total_loss_multiple_batches, \
    specific_loss_multiple_batches


def build_arg_parser():
    """
    :return:    argparse.ArgumentParser() filled with the standard arguments for a training session.
                    Might need to be enhanced for some train_scripts.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='../../data/multitask_dataset.pkl', help='Data path.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--only_nodes', action='store_true', default=False, help='Evaluate only nodes labels.')
    parser.add_argument('--only_graph', action='store_true', default=False, help='Evaluate only graph labels.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    parser.add_argument('--conv_layers', type=int, default=None, help='Graph convolutions')
    parser.add_argument('--variable_conv_layers', type=str, default='N', help='Graph convolutions function name')
    parser.add_argument('--fc_layers', type=int, default=3, help='Fully connected layers in readout')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function to use.')
    parser.add_argument('--print_every', type=int, default=50, help='Print training results every')
    parser.add_argument('--final_activation', type=str, default='LeakyReLu',
                        help='final activation in both FC layers for nodes and S2S for Graph')
    parser.add_argument('--skip', action='store_true', default=False,
                        help='Whether to use the model with skip connections.')
    parser.add_argument('--gru', action='store_true', default=False,
                        help='Whether to use a GRU in the update function of the layers.')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='Whether to use the model with fixed middle convolutions.')
    parser.add_argument('--variable', action='store_true', default=False,
                        help='Whether to have a variable number of comvolutional layers.')
    return parser


# map from names (as passed as parameters) to function determining number of convolutional layers at runtime
VARIABLE_LAYERS_FUNCTIONS = {
    'N': lambda adj: adj.shape[1],
    'N/2': lambda adj: adj.shape[1] // 2,
    '4log2N': lambda adj: int(4 * math.log2(adj.shape[1])),
    '2log2N': lambda adj: int(2 * math.log2(adj.shape[1])),
    '3sqrtN': lambda adj: int(3 * math.sqrt(adj.shape[1]))
}


def execute_train(gnn_args, args):
    """
    :param gnn_args: the description of the model to be trained (expressed as arguments for GNN.__init__)
    :param args: the parameters of the training session
    """
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = 'cuda' if args.cuda else 'cpu'
    print('Using device:', device)

    # load data
    adj, features, node_labels, graph_labels = load_dataset(args.data, args.loss, args.only_nodes, args.only_graph,
                                                            print_baseline=True)

    # model and optimizer
    gnn_args = SimpleNamespace(**gnn_args)

    # compute avg_d on the training set
    if 'avg_d' in gnn_args.first_conv_descr['args'] or 'avg_d' in gnn_args.middle_conv_descr['args']:
        dlist = [torch.sum(A, dim=-1) for A in adj['train']]
        avg_d = dict(lin=sum([torch.mean(D) for D in dlist]) / len(dlist),
                     exp=sum([torch.mean(torch.exp(torch.div(1, D)) - 1) for D in dlist]) / len(dlist),
                     log=sum([torch.mean(torch.log(D + 1)) for D in dlist]) / len(dlist))
    if 'avg_d' in gnn_args.first_conv_descr['args']:
        gnn_args.first_conv_descr['args']['avg_d'] = avg_d
    if 'avg_d' in gnn_args.middle_conv_descr['args']:
        gnn_args.middle_conv_descr['args']['avg_d'] = avg_d

    gnn_args.device = device
    gnn_args.nfeat = features['train'][0].shape[2]
    gnn_args.nodes_out = node_labels['train'][0].shape[-1]
    gnn_args.graph_out = graph_labels['train'][0].shape[-1]
    if gnn_args.variable:
        assert gnn_args.conv_layers is None, "If model is variable, you shouldn't specify conv_layers (maybe you " \
                                             "meant variable_conv_layers?) "
    else:
        assert gnn_args.conv_layers is not None, "If the model is not variable, you should specify conv_layers"
    gnn_args.conv_layers = VARIABLE_LAYERS_FUNCTIONS[
        args.variable_conv_layers] if gnn_args.variable else args.conv_layers
    model = GNN(**vars(gnn_args))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params", pytorch_total_params)

    def move_cuda(dset):
        assert args.cuda, "Cannot move dataset on CUDA, running on cpu"
        if features[dset][0].is_cuda:
            # already on CUDA
            return
        features[dset] = [x.cuda() for x in features[dset]]
        adj[dset] = [x.cuda() for x in adj[dset]]
        node_labels[dset] = [x.cuda() for x in node_labels[dset]]
        graph_labels[dset] = [x.cuda() for x in graph_labels[dset]]

    if args.cuda:
        model.cuda()
        # move train, val to CUDA (delay moving test until needed)
        move_cuda('train')
        move_cuda('val')

    def train(epoch):
        """
        Execute a single epoch of the training loop

        :param epoch:int the number of the epoch being performed (0-indexed)
        """
        t = time.time()

        # train step
        model.train()
        for batch in range(len(adj['train'])):
            optimizer.zero_grad()
            output = model(features['train'][batch], adj['train'][batch])
            loss_train = total_loss(output, (node_labels['train'][batch], graph_labels['train'][batch]), loss=args.loss,
                                    only_nodes=args.only_nodes, only_graph=args.only_graph)
            loss_train.backward()
            optimizer.step()

        # validation epoch
        model.eval()
        output_zip = [model(features['val'][batch], adj['val'][batch]) for batch in range(len(adj['val']))]
        output = ([x[0] for x in output_zip], [x[1] for x in output_zip])

        loss_val = total_loss_multiple_batches(output, (node_labels['val'], graph_labels['val']), loss=args.loss,
                                               only_nodes=args.only_nodes, only_graph=args.only_graph)

        return loss_train.data.item(), loss_val

    def compute_test():
        """
        Evaluate the current model on all the sets of the dataset, printing results.
        This procedure is destructive on datasets.
        """
        model.eval()

        sets = list(features.keys())
        for dset in sets:
            # move data on CUDA if not already on it
            if args.cuda:
                move_cuda(dset)

            output_zip = [model(features[dset][batch], adj[dset][batch]) for batch in range(len(adj[dset]))]
            output = ([x[0] for x in output_zip], [x[1] for x in output_zip])
            loss_test = total_loss_multiple_batches(output, (node_labels[dset], graph_labels[dset]), loss=args.loss,
                                                    only_nodes=args.only_nodes, only_graph=args.only_graph)
            print("Test set results ", dset, ": loss= {:.4f}".format(loss_test))
            print(dset, ": ",
                  specific_loss_multiple_batches(output, (node_labels[dset], graph_labels[dset]), loss=args.loss,
                                                 only_nodes=args.only_nodes, only_graph=args.only_graph))

            # free unnecessary data
            del output_zip
            del output
            del loss_test
            del features[dset]
            del adj[dset]
            del node_labels[dset]
            del graph_labels[dset]
            torch.cuda.empty_cache()

    sys.stdout.flush()
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = -1

    sys.stdout.flush()
    with tqdm(range(args.epochs), leave=True, unit='epoch') as t:
        for epoch in t:
            loss_train, loss_val = train(epoch)
            loss_values.append(loss_val)
            t.set_description('loss.train: {:.4f}, loss.val: {:.4f}'.format(loss_train, loss_val))
            if loss_values[-1] < best:
                # save current model
                torch.save(model.state_dict(), '{}.pkl'.format(epoch))
                # remove previous model
                if best_epoch >= 0:
                    os.remove('{}.pkl'.format(best_epoch))
                # update training variables
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                print('Early stop at epoch {} (no improvement in last {} epochs)'.format(epoch + 1, bad_counter))
                break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    with torch.no_grad():
        compute_test()
