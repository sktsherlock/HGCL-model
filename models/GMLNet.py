import torch
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from poolings.CONPool import CONPool

class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5


class SAGNet(torch.nn.Module):
    def __init__(self, config, args):
        super(SAGNet, self).__init__()
        self.num_features = args.num_features #在data里 
        self.num_classes = args.num_classes #在data里
        self.hidden = config.hidden #yes
        self.pooling_ratio = config.pooling_ratio #yes
        self.dropout = config.dropout #yes


        self.pool1 = CONPool(self.hidden, ratio=self.pooling_ratio) 
        self.pool2 = CONPool(self.hidden, ratio=self.pooling_ratio)
        self.pool3 = CONPool(self.hidden, ratio=self.pooling_ratio)

        self.linear1 = torch.nn.Linear(self.hidden * 2, self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden, self.hidden // 2)
        self.linear3 = torch.nn.Linear(self.hidden // 2, self.num_classes)

        self.dis_loss1, self.dis_loss2, self.dis_loss3 = None, None, None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # NEw
        x = F.relu(self.conv1(x, edge_index)) #得到新的特征矩阵
        x, edge_index, _, batch, _, loss = self.pool1(x, edge_index, None, batch) # edge_index表示A 邻接矩阵
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss1 = loss

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, loss = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss2 = loss

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, loss = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.dis_loss3 = loss
    
        x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=1)


        return x
    
    def compute_disentangle_loss(self):
        return (self.dis_loss1 + self.dis_loss2 + self.dis_loss3) / 3
