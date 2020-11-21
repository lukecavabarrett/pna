import time
import dgl
import torch
from torch.utils.data import Dataset
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.graphproppred import Evaluator
import torch.utils.data


class HIVDGL(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


class HIVDataset(Dataset):
    def __init__(self, name, verbose=True):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.dataset = DglGraphPropPredDataset(name = 'ogbg-molhiv')
        self.split_idx = self.dataset.get_idx_split()

        self.train = HIVDGL(self.dataset, self.split_idx['train'])
        self.val = HIVDGL(self.dataset, self.split_idx['valid'])
        self.test = HIVDGL(self.dataset, self.split_idx['test'])

        self.evaluator = Evaluator(name='ogbg-molhiv')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]