import os.path as osp
import argparse

import torch
from torch import tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T

from ogb.nodeproppred import Evaluator

from model import *
from utils import *

    
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='conv')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--K', type=int, default=10)  ### length of sequence we want to use (K<=P)
parser.add_argument('--P', type=int, default=10)  ### length of path we considered during precomputing (K<=P)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--weight_decay', type=float, default=0.00005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pe_drop', type=float, default=0.5)  ### dropout rate for positional encoding
parser.add_argument('--kernel_size', type=int, default=7)  ### conv kernel size
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--vt_batch_size', type=int, default=10000)  ### batch size used for val/test
parser.add_argument('--log_step', type=int, default=100)
parser.add_argument('--save_precomputed_data_path', type=str, default='./precomputed_data') ### path to precomputed data
args = parser.parse_args()


tab_printer(args)


### Load dataset
print('===Transductive learning===')
data = torch.load(os.path.join(args.save_precomputed_data_path, 'ogbn-products_transductive_P' + str(args.P)+'.pt'))

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_node_features = int(data.sequence_feature.shape[-1])
num_classes = int(max(data.y)+1)
    
if args.model == 'attn': # Neighbor2Seq+Attn w/o PE
    model = ATTNET(args.K, num_node_features, num_classes, args.hidden, args.dropout).to(device)
elif args.model == 'conv': # Neighbor2Seq+Conv
    model = CONVNET(args.K, num_node_features, num_classes, args.hidden, args.kernel_size, args.dropout).to(device)
elif args.model == 'posattn': # Neighbor2Seq+Attn
    model = POSATTN(args.K, num_node_features, num_classes, args.hidden, args.dropout, args.pe_drop).to(device)
else:
    print("Wrong model!")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_dataset = SimpleDataset(data.sequence_feature[data.train_mask], data.y[data.train_mask])
valid_dataset = SimpleDataset(data.sequence_feature[data.val_mask], data.y[data.val_mask])
test_dataset = SimpleDataset(data.sequence_feature[data.test_mask], data.y[data.test_mask])

print("======================================")
print("=====Total number of nodes in ogbn-products:", len(train_dataset)+len(valid_dataset)+len(test_dataset), "=====")
print("=====Total number of training nodes in ogbn-products:", len(train_dataset), "=====")
print("=====Total number of validation nodes in ogbn-products:", len(valid_dataset), "=====")
print("=====Total number of test nodes in ogbn-products:", len(test_dataset), "=====")
print("======================================")

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.vt_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.vt_batch_size, shuffle=False)

print('#Parameters:', sum(p.numel() for p in model.parameters()))

loss_op = torch.nn.CrossEntropyLoss()
    
def train(train_loader, evaluator):
    model.train()
    y_true, y_pred = [], []
    for sequence_feature, y in train_loader:
        sequence_feature, y = sequence_feature.to(device), y.to(device).squeeze()
        optimizer.zero_grad()
        logits = model(sequence_feature)
        loss_op(logits, y).backward()
        optimizer.step()
        pred = logits.argmax(dim=-1, keepdim=True)
        y_pred.append(pred)
        y_true.append(y)
    y_trues = torch.cat(y_true, dim=0).unsqueeze(-1)
    y_preds = torch.cat(y_pred, dim=0)
    return evaluator.eval({
        'y_true': y_trues,
        'y_pred': y_preds,
    })['acc']


@torch.no_grad()
def test(loader, evaluator):
    model.eval()
    y_true, y_pred = [], []
    for sequence_feature, y in loader:
        sequence_feature, y = sequence_feature.to(device), y.to(device).squeeze()
        logits = model(sequence_feature)
        pred = logits.argmax(dim=-1, keepdim=True)
        y_pred.append(pred)
        y_true.append(y)
    y_trues = torch.cat(y_true, dim=0).unsqueeze(-1)
    y_preds = torch.cat(y_pred, dim=0)
    return evaluator.eval({
        'y_true': y_trues,
        'y_pred': y_preds,
    })['acc']





train_results = []
val_results = []
test_results = []
evaluator = Evaluator(name='ogbn-products')
for run in range(1, args.runs+1):
    model.reset_parameters()  ### reset model parameters for each run
    best_val_result = cor_train_results = test_result = best_epoch = 0
    for epoch in range(1, args.epochs+1):
        train_result = train(train_loader, evaluator)
        val_result = test(valid_loader, evaluator)
        if val_result > best_val_result:
            best_val_result = val_result
            test_result = test(test_loader, evaluator)
            best_epoch = epoch 
            cor_train_result = train_result
        if epoch%args.log_step == 0:
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_result, val_result, test_result))
    train_results.append(cor_train_result)
    val_results.append(best_val_result)
    test_results.append(test_result)
    print('================================')
    print('Run:', run, '; Best epoch:', best_epoch, '; Training:', cor_train_result, '; Best validation:', best_val_result, '; Test:', test_result)
print('================================')
train_results, val_results, test_results = tensor(train_results), tensor(val_results), tensor(test_results)
print('Total runs:', args.runs)
print('Training: {:.4f} ± {:.4f}, Best validation: {:.4f} ± {:.4f}, Test: {:.4f} ± {:.4f}'.
          format(train_results.mean().item(), train_results.std().item(), val_results.mean().item(), val_results.std().item(), test_results.mean().item(), test_results.std().item()))
print('Training:', train_results)
print('Best validation:', val_results)
print('Test:', test_results)