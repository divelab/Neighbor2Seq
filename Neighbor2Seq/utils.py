from texttable import Texttable
from torch.utils.data import Dataset
from sklearn import metrics
import torch




class SimpleDataset(Dataset):
    def __init__(self, sequence_feature, y):
        self.sequence_feature = sequence_feature
        self.y = y
        assert self.sequence_feature.size(0) == self.y.size(0)

    def __len__(self):
        return self.sequence_feature.size(0)

    def __getitem__(self, idx):
        return self.sequence_feature[idx], self.y[idx]

    
def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

    
def calc_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="micro")