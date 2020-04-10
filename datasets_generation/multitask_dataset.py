import argparse
import os
import pickle
import sys

import numpy as np
import torch
from inspect import signature

from datasets_generation import graph_algorithms
from datasets_generation.graph_generation import GraphType, generate_graph


class DatasetMultitask:

    def __init__(self, n_graphs, N, seed, graph_type, get_nodes_labels, get_graph_labels, print_every, sssp, filename):
        self.adj = {}
        self.features = {}
        self.nodes_labels = {}
        self.graph_labels = {}
        generated = 0

        def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd=""):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end=printEnd)

        def to_categorical(x, N):
            v = np.zeros(N)
            v[x] = 1
            return v

        for dset in N.keys():
            if dset not in n_graphs:
                n_graphs[dset] = n_graphs['default']

            total_n_graphs = sum(n_graphs[dset])

            set_adj = [[] for _ in n_graphs[dset]]
            set_features = [[] for _ in n_graphs[dset]]
            set_nodes_labels = [[] for _ in n_graphs[dset]]
            set_graph_labels = [[] for _ in n_graphs[dset]]
            generated = 0

            progress_bar(0, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(0, total_n_graphs))

            for batch, batch_size in enumerate(n_graphs[dset]):
                for i in range(batch_size):
                    # generate a random graph of type graph_type and size N
                    seed += 1
                    adj, features, type = generate_graph(N[dset][batch], graph_type, seed=seed)

                    while np.min(np.max(adj, 0)) == 0.0:
                        # remove graph with singleton nodes
                        seed += 1
                        adj, features, _ = generate_graph(N[dset][batch], type, seed=seed)

                    generated += 1
                    if generated % print_every == 0:
                        progress_bar(generated, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                                     suffix='({} of {})'.format(generated, total_n_graphs))

                    # make sure there are no self connection
                    assert np.all(
                        np.multiply(adj, np.eye(N[dset][batch])) == np.zeros((N[dset][batch], N[dset][batch])))

                    if sssp:
                        # define the source node
                        source_node = np.random.randint(0, N[dset][batch])

                    # compute the labels with graph_algorithms; if sssp add the sssp
                    node_labels = get_nodes_labels(adj, features,
                                                   graph_algorithms.all_pairs_shortest_paths(adj, 0)[source_node]
                                                   if sssp else None)
                    graph_labels = get_graph_labels(adj, features)
                    if sssp:
                        # add the 1-hot feature determining the starting node
                        features = np.stack([to_categorical(source_node, N[dset][batch]), features], axis=1)

                    set_adj[batch].append(adj)
                    set_features[batch].append(features)
                    set_nodes_labels[batch].append(node_labels)
                    set_graph_labels[batch].append(graph_labels)

            self.adj[dset] = [torch.from_numpy(np.asarray(adjs)).float() for adjs in set_adj]
            self.features[dset] = [torch.from_numpy(np.asarray(fs)).float() for fs in set_features]
            self.nodes_labels[dset] = [torch.from_numpy(np.asarray(nls)).float() for nls in set_nodes_labels]
            self.graph_labels[dset] = [torch.from_numpy(np.asarray(gls)).float() for gls in set_graph_labels]
            progress_bar(total_n_graphs, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(total_n_graphs, total_n_graphs), printEnd='\n')

        self.save_as_pickle(filename)

    def save_as_pickle(self, filename):
        """" Saves the data into a pickle file at filename """
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.adj, self.features, self.nodes_labels, self.graph_labels), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./data/multitask_dataset.pkl', help='Data path.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--graph_type', type=str, default='RANDOM', help='Type of graphs in train set')
    parser.add_argument('--nodes_labels', nargs='+', default=["eccentricity", "graph_laplacian_features", "sssp"])
    parser.add_argument('--graph_labels', nargs='+', default=["is_connected", "diameter", "spectral_radius"])
    parser.add_argument('--extrapolation', action='store_true', default=False,
                        help='Generated various test sets of dimensions larger than train and validation.')
    parser.add_argument('--print_every', type=int, default=20, help='')
    args = parser.parse_args()

    if 'sssp' in args.nodes_labels:
        sssp = True
        args.nodes_labels.remove('sssp')
    else:
        sssp = False

    # gets the functions of graph_algorithms from the specified datasets
    nodes_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), args.nodes_labels))
    graph_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), args.graph_labels))


    def get_nodes_labels(A, F, initial=None):
        labels = [] if initial is None else [initial]
        for f in nodes_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.swapaxes(np.stack(labels), 0, 1)


    def get_graph_labels(A, F):
        labels = []
        for f in graph_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.asarray(labels).flatten()


    data = DatasetMultitask(n_graphs={'train': [512] * 10, 'val': [128] * 5, 'default': [256] * 5},
                            N={**{'train': range(15, 25), 'val': range(15, 25)}, **(
                                {'test-(20,25)': range(20, 25), 'test-(25,30)': range(25, 30),
                                 'test-(30,35)': range(30, 35), 'test-(35,40)': range(35, 40),
                                 'test-(40,45)': range(40, 45), 'test-(45,50)': range(45, 50),
                                 'test-(60,65)': range(60, 65), 'test-(75,80)': range(75, 80),
                                 'test-(95,100)': range(95, 100)} if args.extrapolation else
                                {'test': range(15, 25)})},
                            seed=args.seed,
                            graph_type=getattr(GraphType, args.graph_type), get_nodes_labels=get_nodes_labels,
                            get_graph_labels=get_graph_labels, print_every=args.print_every, sssp=sssp,
                            filename=args.out)

    data.save_as_pickle(args.out)
