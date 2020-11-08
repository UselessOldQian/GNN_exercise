# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, SGConv
from torch_geometric.utils import dropout_adj

import numpy as np

class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GraphCNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=out_c)

    def forward(self, data):
        # data.x data.edge_index
        x = data.x  # [N, C]
        edge_index = data.edge_index  # [2 ,E]
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        hid = F.relu(hid)
        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c]
        out = F.log_softmax(out, dim=1)  # [N, out_c]

        return out


# todo list
class splineGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(splineGCN, self).__init__()
        self.conv1 = SplineConv(in_c, hid_c, dim=1, kernel_size=2)
        self.conv2 = SplineConv(hid_c, out_c, dim=1, kernel_size=2)

    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class sgcGCN(nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(sgcGCN, self).__init__()
        self.conv1 = SGConv( in_c, hid_c, K=2)
        self.conv2 = SGConv(hid_c, out_c,K=2)
        # self.conv1 = SGConv(in_c, out_c,K=2,cached=True)

    def forward(self,data):
        x = data.x  # [N, C]
        edge_index = data.edge_index
        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        hid = F.relu(hid)
        x = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c]
        # x = self.conv1(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class AGNN(nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(AGNN, self).__init__()
        self.lin1 = torch.nn.Linear(in_c, hid_c)
        self.prop1 = pyg_nn.AGNNConv(requires_grad=False)
        self.prop2 = pyg_nn.AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(hid_c, out_c)

    def forward(self,data):
        x = data.x  # [N, C]
        edge_index = data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class gatGCN(nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(gatGCN, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_c, hid_c, heads=hid_c, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = pyg_nn.GATConv(hid_c * hid_c, out_c, heads=1, concat=True,
                             dropout=0.6)


    def forward(self,data):
        x = data.x  # [N, C]
        edge_index = data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class taGCN(nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(taGCN, self).__init__()
        self.conv1 = pyg_nn.TAGConv(in_c, hid_c)
        self.conv2 = pyg_nn.TAGConv(hid_c, out_c)


    def forward(self,data):
        x = data.x  # [N, C]
        edge_index = data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class GUnet(nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(GUnet, self).__init__()
        pool_ratios = [2000 / 2708, 0.5]
        self.unet = pyg_nn.GraphUNet(in_c, 32, out_c,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self,data):
        x = data.x  # [N, C]
        edge_index = data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        x = F.dropout(data.x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        out = F.log_softmax(x, dim=1)  # [N, out_c]
        return out

class armaGCN(torch.nn.Module):
    def __init__(self, in_c,hid_c,  out_c):
        super(armaGCN,self).__init__()
        self.conv1 = pyg_nn.ARMAConv(in_c, hid_c, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = pyg_nn.ARMAConv(hid_c, out_c, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=None)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class Appnp(torch.nn.Module):
    def __init__(self, in_c, hid_c ,out_c):
        super(Appnp,self).__init__()
        self.lin1 = torch.nn.Linear(in_c,hid_c)
        self.lin2 = torch.nn.Linear(hid_c,out_c)
        self.prop1 = pyg_nn.APPNP(10,0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self,data):
        x, edge_index = data.x , data.edge_index
        x = F.dropout(x, p=0.5, training = self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p =0.5, training = self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
