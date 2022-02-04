import os.path as osp
import os
from texttable import Texttable
import argparse

import torch
from torch_geometric.datasets import Flickr, Reddit, Yelp
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph, to_undirected, dropout_adj
from torch_sparse import fill_diag, SparseTensor
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

from utils import *
import time 
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Flickr')
parser.add_argument('--P', type=int, default=10)  ### length of path we want to consider
parser.add_argument('--transductive', type=bool, default=False)  ### transductive setting or inductive setting
parser.add_argument('--add_self_loop', type=bool, default=False)
parser.add_argument('--dropedge_rate', type=float, default=0.4)  ### applied for ogbn-papers100M
parser.add_argument('--save_path', type=str, default='./precomputed_data')
args = parser.parse_args()



tab_printer(args)



####### Save the whole processesed dataset as one Pytorch Geometric Data

def precompute_transductive(data, P, add_self_loop, is_symmetric, reverse=False):
    to_sp = T.ToSparseTensor()
    data = to_sp(data)
    adj_t = data.adj_t
    if not is_symmetric:
        print('Converting to undirected graph...')
        adj_t = adj_t.to_symmetric()
    if reverse:
        print('Reverse directed edges...')
        adj_t = adj_t.t()
    if add_self_loop:
        print('Adding self loop...')
        adj_t = fill_diag(adj_t, 1)
    assert data.x is not None
    x_tmp = data.x
    print('Precomputing...... (This may take a while)')
    if args.dataset == "ogbn-papers100M":
        xs = []
        print('For ogbn-papers100M, we only need to save the samples with labels.')
        precomputed_sequence_feature = torch.cat([x_tmp[data.train_mask], x_tmp[data.val_mask], x_tmp[data.test_mask]], dim=0)
        new_y = torch.cat([data.y[data.train_mask], data.y[data.val_mask], data.y[data.test_mask]], dim=0)
        n_train = sum(data.train_mask)
        n_val = sum(data.val_mask)
        n_test = sum(data.test_mask)

        new_train_mask = torch.tensor([True]*n_train+[False]*n_val+[False]*n_test)
        new_val_mask = torch.tensor([False]*n_train+[True]*n_val+[False]*n_test)
        new_test_mask = torch.tensor([False]*n_train+[False]*n_val+[True]*n_test)
        xs.append(precomputed_sequence_feature)

        for i in range(1, P + 1):
            print('Computing for P:', i)
            x_tmp = adj_t @ x_tmp
            precomputed_sequence_feature = torch.cat([x_tmp[data.train_mask], x_tmp[data.val_mask], x_tmp[data.test_mask]], dim=0)
            xs.append(precomputed_sequence_feature)
        
        tmp_sequence_feature = torch.stack(xs, dim=1)
        precomputed_data = Data(sequence_feature=tmp_sequence_feature, train_mask=new_train_mask, val_mask=new_val_mask, test_mask=new_test_mask, y=new_y)

    else:
        xs = []
        xs.append(x_tmp)
        for i in range(1, P + 1):
            print('Computing for P:', i)
            x_tmp = adj_t @ x_tmp
            xs.append(x_tmp)
        tmp_sequence_feature = torch.stack(xs, dim=1)
        precomputed_data = Data(sequence_feature=tmp_sequence_feature, train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask, y=data.y) ### sequence_feature's shape: [N, P+1, d]
        
    return precomputed_data



def precompute_inductive(data, P, add_self_loop, is_sorted, is_symmetric):
    assert data.x is not None
    train_edge_index = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0]
    train_x = data.x[data.train_mask]
    N = train_x.shape[0]
    train_adj_t = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(N, N), is_sorted=is_sorted)
    if not is_symmetric:
        train_adj_t = train_adj_t.to_symmetric()
    if add_self_loop:
         train_adj_t = fill_diag(train_adj_t, 1)
    train_xs = []
    x_tmp = train_x
    train_xs.append(x_tmp)
    print('Precomputing for taining nodes...... (This may take a while)')
    for i in range(1, P + 1):
        print('Computing for P:', i)
        x_tmp = train_adj_t @ x_tmp
        train_xs.append(x_tmp)
    train_sequence_feature=torch.stack(train_xs, dim=1)  ###[N_train, P+1, d]
        
    to_sp = T.ToSparseTensor()
    data = to_sp(data)
    adj_t = data.adj_t
    if not is_symmetric:
        adj_t = adj_t.to_symmetric()
    if add_self_loop:
        adj_t = fill_diag(adj_t, 1)
    vt_xs = []
    x_tmp = data.x
    vt_xs.append(x_tmp)
    print('Precomputing for val/test nodes...... (This may take a while)')
    for i in range(1, P + 1):
        print('Computing for P:', i)
        x_tmp = adj_t @ x_tmp
        vt_xs.append(x_tmp)
    val_sequence_feature=torch.stack(vt_xs, dim=1)[data.val_mask] ###[N_val, P+1, d]
    test_sequence_feature=torch.stack(vt_xs, dim=1)[data.test_mask] ###[N_test, P+1, d]
    
    n_train = sum(data.train_mask)
    n_val = sum(data.val_mask)
    n_test = sum(data.test_mask)
    n = n_train + n_val + n_test
    
    train_mask = torch.tensor([True]*n_train+[False]*n_val+[False]*n_test)
    val_mask = torch.tensor([False]*n_train+[True]*n_val+[False]*n_test)
    test_mask = torch.tensor([False]*n_train+[False]*n_val+[True]*n_test)
    precomputed_data = Data(sequence_feature=torch.cat([train_sequence_feature,val_sequence_feature,test_sequence_feature], dim=0), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, y=torch.cat([data.y[data.train_mask], data.y[data.val_mask], data.y[data.test_mask]], dim=0))  ### sequence_feature's shape: [N, P+1, d]
    
    return precomputed_data







### Load dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), './data', args.dataset)
is_sorted = True
is_symmetric = True
if args.dataset == 'Flickr':
    dataset = Flickr(path, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset == 'Reddit':
    dataset = Reddit(path, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset == 'Yelp':
    dataset = Yelp(path, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset == 'ogbn-papers100M':
    dataset = PygNodePropPredDataset('ogbn-papers100M', path)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index, _ = dropout_adj(data.edge_index, p = args.dropedge_rate, num_nodes= data.num_nodes)  ### drop some edges randomly to save computation
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_mask = torch.tensor([False]*(int(data.x.size(0))))
    val_mask = torch.tensor([False]*(int(data.x.size(0))))
    test_mask = torch.tensor([False]*(int(data.x.size(0))))
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    is_sorted = False
    reverse  = True
elif args.dataset == 'ogbn-products':
    dataset = PygNodePropPredDataset('ogbn-products', path)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    train_mask = torch.tensor([False]*(len(train_idx)+len(val_idx)+len(test_idx)))
    val_mask = torch.tensor([False]*(len(train_idx)+len(val_idx)+len(test_idx)))
    test_mask = torch.tensor([False]*(len(train_idx)+len(val_idx)+len(test_idx)))
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    is_sorted = False
    reverse  = False

os.makedirs(args.save_path, exist_ok=True)

if args.transductive:
    print('This precomputed data is used for transductive learning.')
    pre_time_rec = []
    for run in range(1,2):
        ori_data = data
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pre_data = precompute_transductive(ori_data, args.P, args.add_self_loop, is_symmetric, reverse=reverse)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        duration = t_end - t_start
        print(duration)
        pre_time_rec.append(duration)
        
    print("Precomputing time per run:", np.mean(pre_time_rec), "±", np.std(pre_time_rec))
    print('Precomputed data is:', pre_data)
    
    torch.save(pre_data, os.path.join(args.save_path, args.dataset + '_transductive_P' + str(args.P)+'.pt'))
    print('Precomputed data saved!!!')
else:
    print('This precomputed data is used for inductive learning.')
    pre_data = precompute_inductive(data, args.P, args.add_self_loop, is_sorted, is_symmetric)
    print('Precomputed data is:', data)
    torch.save(pre_data, os.path.join(args.save_path, args.dataset + '_inductive_P' + str(args.P)+'.pt'))
    print('Precomputed data saved!!!')