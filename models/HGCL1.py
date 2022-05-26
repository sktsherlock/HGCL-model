from torch_geometric.nn import GCNConv, GATConv, LEConv, SAGEConv, GraphConv, norm
from torch_geometric.data import Data
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
from torch_geometric.utils import softmax, dense_to_sparse, add_remaining_self_loops, remove_self_loops
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm, coalesce
from torch_scatter import scatter_add, scatter
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as GINpool
from utils import Far,PAP1Loss,local_global_loss_, f_adj, global_global_loss_ ,global_global_negative_loss_
from models.Encoder import GIN
from losses.infonce import InfoNCE
from MLP import MLP

class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.9
        self.dropout = 0.5
        self.out = 128

class CONPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, non_lin=torch.tanh):
        super(CONPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.non_lin = non_lin
        self.hidden_dim = in_channels
        self.transform = GraphConv(in_channels, self.hidden_dim)
        self.pp_conv = GraphConv(self.hidden_dim, self.hidden_dim)
        self.np_conv = GraphConv(self.hidden_dim, self.hidden_dim)

        self.l_pooling = GraphConv(self.hidden_dim, 1) 
        self.s_pooling = GraphConv(self.hidden_dim, 1)
  

    def forward(self, x, edge_index, device,  edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))  
        if x is None:
            x = torch.ones((batch.shape[0], 1 )).to(device = device)

        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        #print(x_transform.shape) #维度（形式）保持不变
        x_tl = F.leaky_relu(self.pp_conv(x_transform, edge_index), 0.2)
        #print(x_tp.shape)
        x_ts = F.leaky_relu(self.np_conv(x_transform, edge_index), 0.2)
        #print(x_tn.shape) #都是[点的数量,in_channels数量]  torch.Size([1693, 128])
        s_lp = self.l_pooling(x_tl, edge_index).squeeze()#np.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉，但是对非单维的维度不起作用。
        #print(s_lp.shape) #torch.Size([1693] torch.Size([1409]) torch.Size([1173]) 三次之后
        s_sp = self.s_pooling(x_ts, edge_index).squeeze()#利用squeeze（）函数将表示向量的数组转换为秩为1的数组
        #print(s_sp.shape) #torch.Size([1693]
        #print("--------------------------------------------------")

        perm_positive = topk(s_lp, self.ratio, batch)
        perm_negative = topk(s_sp, self.ratio, batch)
        x_pp = x_transform[perm_positive] * self.non_lin(s_lp[perm_positive]).view(-1, 1) 
        x_np = x_transform[perm_negative] * self.non_lin(s_sp[perm_negative]).view(-1, 1)

        x_pp_readout = GINpool(x_pp, batch[perm_positive])
        x_np_readout = GINpool(x_np, batch[perm_negative]) #gap
        loss = Far.compute(self, x_pp_readout, x_np_readout)
        #loss = global_global_negative_loss_(x_pp_readout, x_np_readout)
        #loss = - InfoNCE.compute(self, x_pp_readout, x_np_readout) #互信息最小化
        score = (s_lp + s_sp)/2#.squeeze()


        perm = topk(score, self.ratio, batch)

        num_nodes = int(x_transform.shape[0])
        x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        #x = x[perm] 本来应该是这样的 
        batch = batch[perm]

        filter_edge_index, filter_edge_attr = f_adj(edge_index, edge_attr = None, perm = perm, num_nodes = num_nodes)

        return x, filter_edge_index, filter_edge_attr, batch, loss ,perm_positive, perm_negative 

class HGCL1(torch.nn.Module):
    def __init__(self, config, args, layers, mid):
        super(HGCL1, self).__init__()
        self.num_features = args.num_features #在data里 
        self.hidden = config.hidden #yes
        self.pooling_ratio = config.pooling_ratio #yes
        self.dropout = config.dropout #yes
        # model = SAGNet(config, args)  输入格式
        self.num_layers = layers 
        self.mid = mid 
        self.inputdim = self.mid * self.num_layers 
        self.out = config.out
        self.conv1 = GIN( self.num_features, self.mid, self.num_layers)


        self.pool1 = CONPool(self.num_features, ratio=self.pooling_ratio) 


        self.projection_head_1 = MLP(in_channels= self.inputdim , hidden_channels= self.hidden, out_channels = self.out)
        

        self.dis_loss1 =0.0
        self.cl_pool1_loss, self.cl_pool2_loss, self.cl_pool3_loss = 0.0, 0.0, 0.0 

    def forward(self, x, edge_index, batch, device):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if x is None:
            x = torch.ones((batch.shape[0], 1 )).to(device = device)
        # NEw
        #输入的没问题

        X_M1, M1_P = self.conv1(x, edge_index, batch) #得到新的特征矩阵 1151,64
        M1_con = self.projection_head_1(M1_P)
        x1, edge_index1, _, batch1, loss1, perm_positive, perm_negative   = self.pool1(x, edge_index, device = device,edge_attr = None, batch = batch) # edge_index表示A 邻接矩阵

        x_l1 = x[perm_positive] 
        batch_l1 = batch[perm_positive]

        l1_A, _  = f_adj(edge_index,edge_attr = None,perm =  perm_positive, num_nodes = int(x.shape[0]))
        X_l1, L1_readout = self.conv1(x_l1, l1_A, batch_l1)
        L1_con = self.projection_head_1(L1_readout)
        x_s1 = x[perm_negative] 
        batch_s1 = batch[perm_negative]

        s1_A, _  = f_adj(edge_index,edge_attr = None, perm = perm_negative,num_nodes = int(x.shape[0]))
        X_s1, S1_readout = self.conv1(x_s1, s1_A, batch_s1)
        S1_con = self.projection_head_1(S1_readout)
        #self.cl_pool1_loss = PAP1Loss.compute(self, anchor = M1_con, sample1 = L1_con, sample2 = S1_con )
        self.cl_pool1_loss = InfoNCE.compute(self, L1_con, S1_con) + InfoNCE.compute(self, M1_con, S1_con)
        #self.cl_pool1_loss = (global_global_loss_(M1_con, S1_con) + global_global_loss_(M1_con, L1_con) ) /2
        self.dis_loss1 = loss1
  
        x = M1_P

        return x, M1_P
    
    def compute_Farloss(self):
        return (self.dis_loss1 )#(self.dis_loss1 + self.dis_loss2 +self.dis_loss3) / 3
    

    
    def compute_inner(self):
        return   (self.cl_pool1_loss)
    #def compute_hier(self):
        return (self.infonce_M1M2 )#+self.infonce_M2M3 +self.infonce_M3M1) 
        #之前后边是PAP1LOSS 前边是infonce
        #return (self.M1M2_gl_loss + self.M2M3_gl_loss + self.M3M1_gl_loss)/3 + (self.YH_loss1+self.YH_loss2+self.YH_loss3)/3
    def get_embeddings(self, device, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                #data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch, device)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)
                
        return ret, y
    
    def get_single_embeddings(self, device, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                #data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                _, x = self.forward(x, edge_index, batch, device)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)
                
        return ret, y