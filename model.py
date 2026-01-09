
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import (cosine_sim,
                   kl_div_matrix,
                   symmetric_kl_divergence_global,
                   symmetric_kl_divergence_sample)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0, activate = 'relu'):
        super(GCNEncoder, self).__init__()
        self.layer1 = GCNConv(in_dim, hid_dim)
        self.layer2 = GCNConv(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, out_dim)
        self.act = activate
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        x = self.layer2(x, edge_index)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer3(x)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        return x


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activate='relu'):
        super(MLPEncoder, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.act = activate
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer1(x)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        return x


class GeneNet(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activate='relu'):
        super(GeneNet, self).__init__()
        self.act = activate
        self.dropout = dropout

        self.gcn = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)
        else:
            x = F.relu(x, inplace=True)

        return x


class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, gene_dim, dropout=0.0, device=torch.device('cuda')):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.gene_dim = gene_dim
        self.dropout = dropout
        self.act = 'leakyrelu'
        self.device = device

        self.GNN_encoder = GCNEncoder(self.in_dim, self.hid_dim, self.hid_dim, self.dropout, self.act)
        self.MLP_encoder = MLPEncoder(self.in_dim, self.hid_dim)

        self.gene_encoder = GeneNet(self.gene_dim, self.out_dim)

        self.projector = nn.Linear(self.hid_dim, self.in_dim)

    def forward(self, x, edge_index, edge_index_g=None):
        x_g = x[0].t()

        x_1 = self.MLP_encoder(x[0])
        x_2 = self.GNN_encoder(x[1], edge_index)

        emb_1 = F.normalize(x_1, dim=1, p=2)
        emb_2 = F.normalize(x_2, dim=1, p=2)

        x_1 = F.dropout(x_1, self.dropout, self.training)
        x_2 = F.dropout(x_2, self.dropout, self.training)

        x_1 = self.projector(x_1)
        x_2 = self.projector(x_2)

        x_g = self.gene_encoder(x_g, edge_index_g)
        x_1 = torch.matmul(x_1, x_g)
        x_2 = torch.matmul(x_2, x_g)

        z_1 = F.normalize(x_1, dim=1, p=2)
        z_2 = F.normalize(x_2, dim=1, p=2)

        return emb_1, emb_2, z_1, z_2


    def cross_view_consistency(self, Z1, Z2, adj):
        n_samples = Z1.size()[0]
        self.sim_matrix_z = cosine_sim(Z1, Z2, device=self.device)
        S = self.sim_matrix_z
        mse_loss = F.mse_loss(S, adj + torch.eye(n_samples, device=self.device), reduction='mean')

        return mse_loss

    def embedding_distribution_alignment(self, x_1, x_2, t=1.0):
        # Global SKL divergence
        skl_global = symmetric_kl_divergence_global(x_1, x_2, t)

        # Cell level SKL divergence
        skl_cell = symmetric_kl_divergence_sample(x_1, x_2, t)
        skl_dev = skl_global + skl_cell

        return skl_dev

    def neighborhood_contrastive_alignment(self, x_1, x_2, adj, t=1.0):
        # Neighborhood distribution alignment using SKL divergence for each node.
        dist_matrix = kl_div_matrix(x_1, x_2, t, device=self.device)
        dist_matrix = (dist_matrix + kl_div_matrix(x_2, x_1, t, device=self.device).t())

        # adj without self loops here.
        n_samples = adj.size(0)
        dist_neighnors = dist_matrix * (adj + torch.eye(n_samples, device=self.device))
        sum_dist_neighnors = torch.sum(dist_neighnors, dim=-1) / (torch.sum(adj, dim=-1) + 1)
        sum_sample_dist = torch.sum(dist_matrix * (1 - torch.eye(n_samples, device=self.device)), dim=-1) / (n_samples - 1)
        loss_ndc = torch.mean(sum_dist_neighnors / sum_sample_dist)

        return loss_ndc
