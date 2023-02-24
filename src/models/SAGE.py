import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SAGE, self).__init__()
        self.body = SAGE_Body(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid,nclass)

    def forward(self, g, x):
        x = self.body(g,x)
        x = self.fc(x)
        return x

# def GCN(nn.Module):
class SAGE_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SAGE_Body, self).__init__()

        self.gc1 = SAGEConv(nfeat, nhid, 'mean')
        self.gc2 = SAGEConv(nhid, nhid, 'mean')
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)
        # x = self.dropout(x)
        return x    




