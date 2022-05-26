from re import S
from torch_geometric.nn import   TopKPooling, dense_diff_pool
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from utils import global_global_loss_, global_global_negative_loss_,f_adj, local_global_loss_, YHloss
from MLP import MLP
from models.Encoder import GIN

EPS = 1e-15

class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5
        self.out = 128


def fuse_x(perm1, perm2, batch1, x1, x2, o_batch, edge_index, edge_attr):
    device = x1.device
    dtype = x1.dtype

    perm1 = perm1.cpu().detach().numpy().tolist()
    perm2 = perm2.cpu().detach().numpy().tolist()

    x1 = x1.cpu().detach().numpy().tolist()
    x2 = x2.cpu().detach().numpy().tolist()

    def judge_index(i):
        if i==1:
            x3.append(x1[ls[point][2]])
        else:
            x3.append(x2[ls[point][2]])

    ls=[]
    perm = []
    l=len(perm1)
    x3=[]
    for i in range(l):
        ls.append([perm1[i],1,i])
        ls.append([perm2[i],2,i])
    ls=sorted(ls,key=lambda x:[x[0],x[1]])
    l2=len(ls)
    point=0
    while point<l2:
        try :
            if ls[point][0]==ls[point+1][0] : #and ls[point][1]==ls[point+1][1]
                newls=[]
                a=x1[ls[point][2]]
                b=x2[ls[point][2]]
                for i in range(len(a)):
                    newls.append((a[i]+b[i])/2)
                x3.append(newls)
                perm.append(ls[point][0])
                point+=2
            else:
                i=ls[point][1]
                perm.append(ls[point][0])
                judge_index(i)
                point+=1
        except IndexError:
            #print(point)
            i = ls[point][1]
            perm.append(ls[point][0])
            judge_index(i)
            break

    x3 = np.array(x3)
    x3 = torch.from_numpy(x3)
    x3 = torch.tensor(x3, device= device, dtype= dtype)
    perm = torch.from_numpy(np.array(perm)).to(device)
    new_batch = o_batch[perm].to(device)
    edge_index, edge_attr = f_adj(edge_index, edge_attr, perm)
    edge_index.to(device)
    return x3, new_batch, edge_index, edge_attr

class CPNet(torch.nn.Module):
    def __init__(self, config, args, batch_size, layers, mid,  in_channels=128, ratio=0.6, non_lin=torch.tanh):
        super(CPNet, self).__init__()
        self.num_features = args.num_features  
        self.num_classes = args.num_classes 
        self.hidden = config.hidden 
        self.mid = mid
        self.out = config.out
        self.pooling_ratio = config.pooling_ratio 
        self.dropout = config.dropout 
        self.num_nodes =  batch_size
        self.device = args.device
        self.in_channels = in_channels
        self.w4, self.w5 = 0.8, 0.5
        self.ratio = ratio
        self.ratio2 = self.ratio * self.w4
        self.ratio3 = self.ratio2 * self.w5
        self.non_lin = non_lin
        self.num_layers = layers
        self.inputdim = self.num_layers * self.mid

        self.w1, self.w2, self.w3 = 0.8, 0.2, 0.1
        self.w6, self.w7, self.w8 = 1, 1, 0.5

        self.M1 = GIN( self.num_features, self.mid, self.num_layers) #encoder
        self.LP1  = TopKPooling(in_channels = self.inputdim, ratio = self.ratio ) # self.num_features
        self.SP1 = TopKPooling(in_channels = self.inputdim, ratio = self.ratio ) # 


        # 损失定义
        self.L1M1_loss = 0.
        self.S1M1_loss = 0.
        self.L1S1_loss = 0.
        self.CON_LOSS1 = 0.
        self.YH_loss   = 0.

        self.M1M2_loss = 0.
        self.M2M3_loss = 0.
        self.M3M1_loss = 0.
        self.M1M2_gl_loss = 0.

        self.projection_head_local  = MLP(in_channels= self.inputdim , hidden_channels= self.hidden, out_channels = self.out) # self.num_feature
        self.projection_head_message = MLP(in_channels= self.inputdim , hidden_channels= self.hidden, out_channels = self.out)
        self.projection_head_semantic = MLP(in_channels= self.inputdim , hidden_channels= self.hidden, out_channels = self.out)
        
        
    def forward(self, x, edge_index, batch, edge_attr=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # 第一层
        X_M1, M1_P = self.M1(x, edge_index, batch) #全部的， 池化后的
        #M1_readout = gap(X_M1, batch)
        M1_con = self.projection_head_message(M1_P)
        
        # localpooling 1 
        X_L1, edge_index_local1, edge_attr_local1, batch_l1, perm1_1, _ = self.LP1(X_M1, edge_index, edge_attr = None, batch = batch, attn = None)
        L1_readout = gap(X_L1, batch_l1)
        L1_con = self.projection_head_local(L1_readout)

        X_S1, edge_index_sem1, edge_attr_sem2, batch_s1, perm2_1, _ = self.SP1(X_M1, edge_index, edge_attr = None, batch = batch, attn = None)
        S1_readout = gap(X_S1, batch_s1) 
        S1_con = self.projection_head_semantic(S1_readout)

        #产生第一层的对比损失
        # self.L1M1_loss = global_global_loss_(M1_con, L1_con)
        # self.S1M1_loss = global_global_loss_(M1_con, S1_con)
        # self.L1S1_loss = global_global_negative_loss_(L1_con, S1_con)

        self.CON_LOSS1 = self.L1M1_loss + self.S1M1_loss + self.L1S1_loss

        self.YH_loss = YHloss.compute(self, anchor = M1_con, sample1 = L1_con, sample2 = S1_con )

        z1 = torch.cat((L1_readout, S1_readout), 1) # z1 = torch.cat((M1_P, (L1_readout + S1_readout)/2), 1)
        self.z = torch.cat((L1_readout, S1_readout), 1)
        #fusion 产生下一层的输入
        self.x1, self.batch1, self.edge_index1, self.edge_attr1 = fuse_x(perm1_1, perm2_1, batch_l1, X_L1, X_S1, batch, edge_index, edge_attr)
        # local_global loss
        # self.M1M2_gl_loss = local_global_loss_(X_M1, M2_P, batch)
        # self.M2M3_gl_loss = local_global_loss_(X_M2, M3_P, batch1)
        # self.M3M1_gl_loss = local_global_loss_(X_M3, M1_P, batch2)
        return self.w1 * self.YH_loss

    # def compute_conloss(self):
    #     return (self.w1 * self.YH_loss )

    # def hierarchical_loss(self):
    #     return (self.w6 * self.M1M2_loss + self.w7 * self.M2M3_loss  + self.w8 * self.M3M1_loss )

    def compute_gl_loss(self):
        return (self.M1M2_gl_loss + self.M2M3_gl_loss + self.M3M1_gl_loss)

    def get_embeddings(self): 
        return self.z
    
    def get_newAX(self):
        return self.x1, self.batch1, self.edge_index1, self.edge_attr1

class CONNet(torch.nn.Module):
    def __init__(self, config, args) -> None:
        super(CONNet, self).__init__()
        self.num_features = args.num_features
        self.hidden = config.hidden
        self.pooling_ratio = config.pooling_ratio
        self.conv1 = CPNet(self.num_features, self.hidden)
        self.conv2 = CPNet(self.hidden, self.hidden)
        self.conv3 = CPNet(self.hidden, self.hidden)
        
