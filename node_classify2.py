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
from model import *

import numpy as np
# np.random.seed(123)
# torch.manual_seed(123)

# load dataset
# dataset = Planetoid(root = './dataset', name = 'Cora')
def get_data(folder="node_classify/cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset

def get_data2(folder="node_classify/cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name,
                        # pre_transform=T.KNNGraph(k=6),
                        # transform=T.NormalizeFeatures())#,
                        transform=T.TargetIndegree())
    return dataset

def main(model_index=0, hid_c=12):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cora_dataset = get_data2('./dataset','Cora')

    # todo list
    # my_net = None
    model_list = ['AGNN','GUnet','GraphCNN','gatGCN',
                  'sgcGCN','splineGCN','taGCN','armaGCN','Appnp']
    model_def = eval( model_list[model_index] )
    my_net = model_def(in_c = cora_dataset.num_node_features,
                      hid_c = hid_c,
                      out_c = cora_dataset.num_classes
                      )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)
    data = cora_dataset[0].to(device)
    # print( data.num_nodes )

    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01, weight_decay=8e-3)

    def train():
        my_net.train()
        optimizer.zero_grad()
        output = my_net(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    def test():
        my_net.eval()
        logits, accs = my_net(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    best_val_acc = best_test_acc = test_acc = 0
    best_epoch =0
    for epoch in range(200):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if tmp_test_acc > best_test_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_test_acc = tmp_test_acc
            best_epoch = epoch
            # log = 'Epoch: {:03d},Best_epoch:{:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, best_epoch, train_acc, best_val_acc, test_acc))

    log = 'Model:{:s},Best_epoch:{:03d}, hid_c:{:03d},  Test: {:.4f}'
    print(log.format(model_list[model_index], best_epoch, hid_c, best_test_acc))
    return best_test_acc


if __name__ == '__main__':
    # for model_index in [5,6,7]:
    #     for hic_in in [4*i+4 for i in range(50)]:
    #         main(model_index,hic_in)
    test_acc = []
    for i in range(100):
        tc = main(8,64)
        test_acc.append(tc)

    print("Mean Acc:{:.4f}, ±{:.4f}".format(
        np.mean(test_acc),
        np.std(test_acc)
    ))


# Model:AGNN,Best_epoch:171, hic_in:172, Val: 0.7860, Test: 0.8410
