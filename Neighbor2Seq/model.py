import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ModuleList, Conv1d
import math
from torch.autograd import Variable


class ATTNET(torch.nn.Module):
    def __init__(self, K, num_node_features, num_classes, hidden, dropout):
        super(ATTNET, self).__init__()
        self.lins = ModuleList()
        self.norms = ModuleList()
        for _ in range(K + 1):
            self.lins.append(Linear(num_node_features, hidden))
            self.norms.append(LayerNorm(hidden))
        self.lin = Linear(hidden, num_classes)
        self.proj = Linear(hidden, 1)
        self.K = K
        self.dropout = dropout

    def forward(self, sequence_feature):
        xs = sequence_feature[:, :self.K+1, :]   ### xs: [batch_size, K, d]
        outs = []
        for i, lin in enumerate(self.lins):
            out = F.relu(self.norms[i](lin(xs[:, i, :])))
            out = F.dropout(out, p=self.dropout, training=self.training)
            outs.append(out)
            
        pps = torch.stack(outs, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = F.softmax(retain_score, dim=-1)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        x = self.lin(out)
        return x
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.lin.reset_parameters()
        self.proj.reset_parameters()
    
    
    
    
class CONVNET(torch.nn.Module):
    def __init__(self, K, num_node_features, num_classes, hidden, kernel_size, dropout):
        super(CONVNET, self).__init__()
        self.lins = ModuleList()
        self.norms = ModuleList()
        for _ in range(K + 1):
            self.lins.append(Linear(num_node_features, hidden))
            self.norms.append(LayerNorm(hidden))
        self.conv1 = Conv1d(hidden, hidden, kernel_size, padding=int((kernel_size-1)/2))
        self.conv2 = Conv1d(hidden, hidden, kernel_size, padding=int((kernel_size-1)/2))
        self.lin = Linear(hidden, num_classes)
        self.K = K
        self.dropout = dropout

    def forward(self, sequence_feature):
        xs = sequence_feature[:, :self.K+1, :]   ### xs: [batch_size, K, d]
        outs = []
        for i, lin in enumerate(self.lins):
            out = F.relu(self.norms[i](lin(xs[:, i, :])))
            out = F.dropout(out, p=self.dropout, training=self.training)
            outs.append(out)
            
        pps = torch.stack(outs, dim=2)
        out = self.conv2(F.dropout(F.relu(self.conv1(pps)), p=self.dropout, training=self.training))
        out = torch.mean(out, dim=2)
        x = self.lin(out)
        return x
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()
    

    
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=11):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x_batch):
        x_batch = self.dropout(x_batch + Variable(self.pe[:x_batch[0].size(0), :], requires_grad=False))
        return x_batch
    
    
    
class POSATTN(torch.nn.Module):
    def __init__(self, K, num_node_features, num_classes, hidden, dropout, pe_drop):
        super(POSATTN, self).__init__()
        self.lins = ModuleList()
        self.norms = ModuleList()
        self.pos_encoder = PositionalEncoding(hidden, pe_drop, max_len=K+1)
        for _ in range(K + 1):
            self.lins.append(Linear(num_node_features, hidden))
            self.norms.append(LayerNorm(hidden))
        self.lin = Linear(hidden, num_classes)
        self.proj = Linear(hidden, 1)
        self.K = K
        self.dropout = dropout

    def forward(self, sequence_feature):
        xs = sequence_feature[:, :self.K+1, :]   ### xs: [batch_size, K, d]
        outs = []
        for i, lin in enumerate(self.lins):
            out = F.relu(self.norms[i](lin(xs[:, i, :])))
            out = F.dropout(out, p=self.dropout, training=self.training)
            outs.append(out)
            
        pps = torch.stack(outs, dim=1)
        pps = self.pos_encoder(pps)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = F.softmax(retain_score, dim=-1)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        x = self.lin(out)
        return x
    
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.lin.reset_parameters()
        self.proj.reset_parameters()