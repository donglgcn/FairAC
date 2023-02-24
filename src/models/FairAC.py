import random

import torch.nn as nn
from .GCN import GCN,GCN_Body
from .GAT import GAT,GAT_body
from .SAGE import SAGE_Body
from .HGNN_AC import HGNN_AC
import torch
import torch.nn.functional as F

import numpy as np

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    elif args.model == "SAGE":
        model = SAGE_Body(nfeat, args.num_hidden, args.dropout)
    else:
        print("Model not implement")
        return

    return model


class GNN(nn.Module):

    def __init__(self, nfeat, args):
        super(GNN, self).__init__()

        nhid = args.num_hidden
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, g, x):
        z = self.GNN(g, x)
        y = self.classifier(z)
        return z, y


class FairGnn(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGnn, self).__init__()

        nhid = args.num_hidden
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        self.classifierSen = nn.Linear(nhid, 1)
        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_S = torch.optim.Adam(self.classifierSen.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self, g, x):
        z = self.GNN(g, x)
        y = self.classifier(z)
        s = self.classifierSen(z)
        return z, y, s


# baseAC, used autoencoder to improve performance.
class BaseAC(nn.Module):
    def __init__(self, feature_dim, transformed_feature_dim,  emb_dim, args):
        super(BaseAC, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, transformed_feature_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.fcdecoder = torch.nn.Linear(transformed_feature_dim, feature_dim)
        nn.init.xavier_normal_(self.fcdecoder.weight, gain=1.414)
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                          activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)
        AC_params = list(self.fc.parameters()) + list(self.fcdecoder.parameters()) + list(self.hgnn_ac.parameters())
        self.optimizer_AC = torch.optim.Adam(AC_params, lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        transformed_features = self.fc(feature_src)
        feature_src_re = self.hgnn_ac(bias,
                                 emb_dest, emb_src,
                                 transformed_features)
        feature_hat = self.fcdecoder(transformed_features)
        return feature_src_re, feature_hat

    def feature_transform(self, features):
        return self.fc(features)

    def feature_decoder(self, transformed_features):
        return self.fcdecoder(transformed_features)

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(self.fc(origin_feature), AC_feature, 2).mean()


class FairAC2(nn.Module):
    def __init__(self, feature_dim, transformed_feature_dim, emb_dim, args):
        super(FairAC2, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, 2*transformed_feature_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(2*transformed_feature_dim, transformed_feature_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        self.encoder = torch.nn.Sequential(self.fc, self.relu, self.fc2)

        self.fcdecoder = torch.nn.Linear(transformed_feature_dim, transformed_feature_dim*2)
        self.relu2 = torch.nn.ReLU()
        self.fcdecoder2 = torch.nn.Linear(transformed_feature_dim*2, feature_dim)
        nn.init.xavier_normal_(self.fcdecoder.weight, gain=1.414)
        nn.init.xavier_normal_(self.fcdecoder2.weight, gain=1.414)
        self.decoder = torch.nn.Sequential(self.fcdecoder, self.relu2, self.fcdecoder2)
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                               activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)
        AC_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.hgnn_ac.parameters())
        self.optimizer_AC = torch.optim.Adam(AC_params, lr=args.lr, weight_decay=args.weight_decay)

        # divide AC_params into two parts.
        AE_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer_AE = torch.optim.Adam(AE_params, lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_AConly = torch.optim.Adam(self.hgnn_ac.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.classifierSen = nn.Linear(transformed_feature_dim, args.num_sen_class)
        self.optimizer_S = torch.optim.Adam(self.classifierSen.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        transformed_features = self.encoder(feature_src)
        feature_src_re = self.hgnn_ac(bias,
                                      emb_dest, emb_src,
                                      transformed_features)
        feature_hat = self.decoder(transformed_features)
        return feature_src_re, feature_hat, transformed_features

    def sensitive_pred(self, transformed_features):
        return self.classifierSen(transformed_features)

    def feature_transform(self, features):
        return self.encoder(features)

    def feature_decoder(self, transformed_features):
        return self.decoder(transformed_features)

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(self.encoder(origin_feature).detach(), AC_feature, 2).mean()


class AverageAC(nn.Module):
    def __init__(self):
        super(AverageAC, self).__init__()

    def forward(self, adj, feature_src):
        degree = [max(1,adj[i].sum().item()) for i in range(adj.shape[0])]
        mean_adj = torch.stack([adj[i]/degree[i] for i in range(adj.shape[0])])
        feature_src_re = mean_adj.matmul(feature_src)
        return feature_src_re
