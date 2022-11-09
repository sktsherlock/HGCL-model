import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
from MLP import MLP
from torch import nn
from models.Encoder import GIN

class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5


def innercl(proj_1, proj_2):
    from losses.infonce import infonce
    return infonce(proj_1, proj_2)


class HNet(torch.nn.Module):
    def __init__(self, config, args):
        super(HNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.num_layers = args.num_layers
        self.hidden = config.hidden
        self.pooling_ratio = config.pooling_ratio
        self.dropout = config.dropout

        self.conv1 = GIN(self.num_features, self.hidden, self.num_layers)
        self.conv2 = GIN(self.hidden, self.hidden, self.num_layers)
        self.conv3 = GIN(self.hidden, self.hidden, self.num_layers)

        self.projection_head_1 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_2 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_3 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)

        self.pool1 = TopKPooling(self.hidden, ratio=self.pooling_ratio)
        self.pool2 = TopKPooling(self.hidden, ratio=self.pooling_ratio)
        self.pool3 = TopKPooling(self.hidden, ratio=self.pooling_ratio)

        self.linear1 = torch.nn.Linear(self.hidden * 2, self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden, self.hidden // 2)
        self.linear3 = torch.nn.Linear(self.hidden // 2, self.num_classes)

    def forward(self, data):
        x_0, edge_index_0, batch_0 = data.x, data.edge_index, data.batch
        edge_attr_0 = None

        x, p = self.conv1(x_0, edge_index_0, batch_0)
        proj_1 = self.projection_head_1(p)

        x_1, edge_index_1, edge_attr_1, batch_1, _, _  = self.pool1(x, edge_index_0, edge_attr_0, batch_0)

        g1 = torch.cat([gmp(x_1, batch_1), gap(x_1, batch_1)], dim=1)

        x, p = self.conv2(x_1, edge_index_1, batch_1)
        proj_2 = self.projection_head_2(p)

        x_2, edge_index_2, edge_attr_2, batch_2, _, _  = self.pool2(x, edge_index_1, edge_attr_1, batch_1)
        g2 = torch.cat([gmp(x_2, batch_2), gap(x_2, batch_2)], dim=1)

        x, p = self.conv3(x_2, edge_index_2, batch_2)
        proj_3 = self.projection_head_3(p)
        x_3, edge_index_3, edge_attr_3, batch_3, _, _ = self.pool3(x, edge_index_2, edge_attr_2, batch_2)
        g3 = torch.cat([gmp(x_3, batch_3), gap(x_3, batch_3)], dim=1)

        x = F.relu(g1) + F.relu(g2) + F.relu(g3)

        # x = F.relu(self.linear1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.relu(self.linear2(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.log_softmax(self.linear3(x), dim=-1)

        return x, g1, g2, g3, proj_1, proj_2, proj_3

    def get_embeddings(self, device, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                # data = data[0]
                data.to(device)
                x, *_ = self(data)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)

        return ret, y

#x_1, edge_index_1, edge_attr_1, batch_1, x_2, edge_index_2, edge_attr_2, batch_2, x_3, edge_index_3, edge_attr_3, batch_3
