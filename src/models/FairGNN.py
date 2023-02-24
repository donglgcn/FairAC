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

class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        s = self.estimator(g,x)
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y,s
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ### update E, G
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        s = self.estimator(g,x)
        h = self.GNN(g,x)
        y = self.classifier(h)



        s_g = self.adv(h)

        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()
        y_score = torch.sigmoid(y)
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))


        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        self.adv_loss = self.criterion(s_g,s_score)
        
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()
        self.optimizer_G.step()

        ## update Adv
        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(h.detach())
        self.A_loss = self.criterion(s_g,s_score)
        self.A_loss.backward()
        self.optimizer_A.step()


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


# only has a attention attribute completion model, without autoencoder.
class ClassicAC(nn.Module):
    def __init__(self, emb_dim, args):
        super(ClassicAC, self).__init__()

        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                          activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)
        self.optimizer_AC = torch.optim.Adam(self.hgnn_ac.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        feature_src_re = self.hgnn_ac(bias,
                                 emb_dest, emb_src,
                                 feature_src)
        return feature_src_re, None

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(origin_feature, AC_feature, 2).mean()


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

# Fair AC using Fair select approach. done
class FairSelectAC(nn.Module):
    def __init__(self, feature_dim, transformed_feature_dim,  emb_dim, args):
        super(FairSelectAC, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, transformed_feature_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.fcdecoder = torch.nn.Linear(transformed_feature_dim, feature_dim)
        nn.init.xavier_normal_(self.fcdecoder.weight, gain=1.414)
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=args.attn_vec_dim, dropout=args.dropout,
                          activation=F.elu, num_heads=args.num_heads, cuda=args.cuda)
        AC_params = list(self.fc.parameters()) + list(self.fcdecoder.parameters()) + list(self.hgnn_ac.parameters())
        self.optimizer_AC = torch.optim.Adam(AC_params, lr=args.lr, weight_decay=args.weight_decay)

    def forward(self, bias, emb_dest, emb_src, feature_src, fairadj = False):
        if not fairadj:
            fair_adj = self.fair_select(bias,feature_src)
        else:
            fair_adj = bias
        transformed_features = self.fc(feature_src)
        feature_src_re = self.hgnn_ac(fair_adj,
                                 emb_dest, emb_src,
                                 transformed_features)
        feature_hat = self.fcdecoder(transformed_features)
        return feature_src_re, feature_hat

    def fair_select(self, adj, feature_with_sens):
        sens = feature_with_sens[:,-1] + 1      # covert 0 to 1, 1 to 2. in case adj is 0 which cause wrong counter.
        sens_num_class = len(torch.unique(sens))
        for idx,row in enumerate(adj):
            sens_counter = [0] * (sens_num_class+1)
            sen_row = (row*sens).long()
            sen_row_array = np.array(sen_row.cpu())
            for i in range(sens_num_class+1):
                sens_counter[i] = np.count_nonzero(sen_row_array == i)
            # for i in sen_row:
            #     sens_counter[i] += 1
            sens_counter.remove(sens_counter[0])    # ignore 0, which means the number of no edges nodes pairs
            # fint the min sens_counter that greater than 0
            least_num_sens_class = max(sens_counter)
            for counter in sens_counter:
                if counter > 0 and counter < least_num_sens_class:
                    least_num_sens_class = counter

            remove_number = [max(counter - least_num_sens_class,0) for counter in sens_counter]    # number of edges per class that need to remove to keep fair
            for i,number in enumerate(remove_number):
                if(number > 0):
                    sen_class = i+1
                    sens_idx = np.where(sen_row.cpu() == sen_class)[0]
                    drop_idx = torch.tensor(random.sample(list(sens_idx), number)).long()
                    adj[idx][drop_idx] = 0
        return adj

    def feature_transform(self, features):
        return self.fc(features)

    def feature_decoder(self, transformed_features):
        return self.fcdecoder(transformed_features)

    def loss(self, origin_feature, AC_feature):
        return F.pairwise_distance(self.fc(origin_feature), AC_feature, 2).mean()


class FairAC_GNN(nn.Module):

    def __init__(self, nfeat,transformed_feature_dim,emb_dim, args):
        super(FairAC_GNN, self).__init__()

        nhid = args.num_hidden
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)
        self.classifierSen = nn.Linear(nhid, 1)
        self.ACmodel = BaseAC(nfeat,transformed_feature_dim, emb_dim,args)
        G_params = list(self.ACmodel.parameters()) + list(self.GNN.parameters()) + list(self.classifier.parameters())
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

    def feature_transform(self, features):
        return self.ACmodel.feature_transform(features)

    def feature_decoder(self, transformed_features):
        return self.ACmodel.feature_decoder(transformed_features)
