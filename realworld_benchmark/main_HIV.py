import numpy as np
import os
import time
import random
import argparse, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.HIV_graph_classification.pna_net import PNANet
from data.HIV import HIVDataset  # import dataset
from train.train_HIV_graph_classification import train_epoch_sparse as train_epoch, \
    evaluate_network_sparse as evaluate_network


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def view_model_param(net_params):
    model = PNANet(net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('PNA Total parameters:', total_param)
    return total_param


def train_val_pipeline(dataset, params, net_params):
    t0 = time.time()
    per_epoch_time = []

    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = PNANet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_ROCs, epoch_val_ROCs, epoch_test_ROCs = [], [], []

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate,
                              pin_memory=True)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate,
                            pin_memory=True)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate,
                             pin_memory=True)

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs']), unit='epoch') as t:
            for epoch in t:
                if epoch == -1:
                    model.reset_params()

                t.set_description('Epoch %d' % epoch)
                start = time.time()

                epoch_train_loss, epoch_train_roc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                epoch_val_loss, epoch_val_roc = evaluate_network(model, device, val_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_ROCs.append(epoch_train_roc.item())
                epoch_val_ROCs.append(epoch_val_roc.item())

                _, epoch_test_roc = evaluate_network(model, device, test_loader, epoch)
                epoch_test_ROCs.append(epoch_test_roc.item())

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_ROC=epoch_train_roc.item(), val_ROC=epoch_val_roc.item(),
                              test_ROC=epoch_test_roc.item(), refresh=False)

                per_epoch_time.append(time.time() - start)
                scheduler.step(-epoch_val_roc.item())

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

                print('')

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    best_val_epoch = np.argmax(np.array(epoch_val_ROCs))
    best_train_epoch = np.argmax(np.array(epoch_train_ROCs))
    best_val_roc = epoch_val_ROCs[best_val_epoch]
    best_val_test_roc = epoch_test_ROCs[best_val_epoch]
    best_val_train_roc = epoch_train_ROCs[best_val_epoch]
    best_train_roc = epoch_train_ROCs[best_train_epoch]

    print("Best Train ROC: {:.4f}".format(best_train_roc))
    print("Best Val ROC: {:.4f}".format(best_val_roc))
    print("Test ROC of Best Val: {:.4f}".format(best_val_test_roc))
    print("Train ROC of Best Val: {:.4f}".format(best_val_train_roc))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', type=int, help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--expid', help='Experiment id.')
    parser.add_argument('--aggregators', type=str, help='Aggregators to use.')
    parser.add_argument('--scalers', type=str, help='Scalers to use.')
    parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.')

    args = parser.parse_args()
    print(args.config)

    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    # dataset, out_dir
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = HIVDataset(DATASET_NAME)

    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.aggregators is not None:
        net_params['aggregators'] = args.aggregators
    if args.scalers is not None:
        net_params['scalers'] = args.scalers
    if args.posttrans_layers is not None:
        net_params['posttrans_layers'] = args.posttrans_layers

    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                   dataset.train.graph_lists])
    net_params['avg_d'] = dict(lin=torch.mean(D),
                               exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                               log=torch.mean(torch.log(D + 1)))

    net_params['total_param'] = view_model_param(net_params)
    train_val_pipeline(dataset, params, net_params)


main()
