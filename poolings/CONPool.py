from torch_geometric.nn import   TopKPooling, dense_diff_pool
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv
import numpy as np
from utils import global_global_loss_, global_global_negative_loss_,f_adj, local_global_loss_, PAPloss, PAP1Loss
from MLP import MLP
from models.Encoder import GIN, GCN 

EPS = 1e-15

class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5
        self.out = 128


def fuse_x(perm1, perm2, batch1,x1, x2, o_batch, edge_index, edge_attr):
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
            if ls[point][0]==ls[point+1][0]: # and ls[point][1]==ls[point+1][1]
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
            i = ls[point][1]
            judge_index(i)
            perm.append(ls[point][0])
            break

    x3 = np.array(x3)
    x3 = torch.from_numpy(x3)
    x3 = torch.tensor(x3, device= device, dtype= dtype)
    perm = torch.from_numpy(np.array(perm)).to(device)
    new_batch = o_batch[perm].to(device)
    edge_index, edge_attr = f_adj(edge_index, edge_attr, perm)
    edge_index.to(device)
    return x3, new_batch, edge_index, edge_attr

class CONPool(torch.nn.Module):
    def __init__(self, config, args, batch_size, layers, mid,  in_channels=128, ratio=0.6, non_lin=torch.tanh):
        super(CONPool, self).__init__()
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
        
        self.M1 = GIN( self.num_features, self.mid, self.num_layers)
        #self.GCN = GCNConv(self.num_features, self.hidden)#
        self.LP1  = TopKPooling(in_channels = self.num_features, ratio = self.ratio ) # 
        self.SP1 = TopKPooling(in_channels = self.num_features, ratio = self.ratio ) # 

        self.M2 = GIN(self.num_features, self.mid, self.num_layers)
        self.LP2  = TopKPooling(in_channels = self.num_features, ratio = self.ratio2 ) # 
        self.SP2 = TopKPooling(in_channels = self.num_features, ratio = self.ratio2 ) # 

        self.M3 = GIN( self.num_features, self.mid, self.num_layers)
        self.LP3  = TopKPooling(in_channels = self.num_features, ratio = self.ratio3 ) # 
        self.SP3 = TopKPooling(in_channels = self.num_features, ratio = self.ratio3 ) # 

        # 损失定义
        self.L1M1_loss = 0.
        self.S1M1_loss = 0.
        self.L1S1_loss = 0.
        self.CON_LOSS1 = 0.

        self.L2M2_loss = 0.
        self.S2M2_loss = 0.
        self.L2S2_loss = 0.
        self.CON_LOSS2 = 0.

        self.L3M3_loss = 0.
        self.S3M3_loss = 0.
        self.L3S3_loss = 0.
        self.CON_LOSS3 = 0.
        self.M1M2_loss = 0.
        self.M2M3_loss = 0.
        self.M3M1_loss = 0.
        self.M1M2_gl_loss = 0.
        self.M2M3_gl_loss = 0.
        self.M3M1_gl_loss = 0.
        self.YH_loss1 = 0.
        self.YH_loss2 = 0.
        self.YH_loss3 = 0.
        self.H_loss = 0.

        self.projection_head_local  = MLP(in_channels= self.num_features , hidden_channels= self.hidden, out_channels = self.out)
        self.projection_head_message = MLP(in_channels= self.inputdim , hidden_channels= self.hidden, out_channels = self.out)
        self.projection_head_semantic = MLP(in_channels= self.num_features , hidden_channels= self.hidden, out_channels = self.out)

        
    def forward(self, x, edge_index, batch, device, edge_attr=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if x is None:
            x = torch.ones((batch.shape[0], 1 )).to(device = device)

        # 第一层
        X_M1, M1_P = self.M1(x, edge_index, batch)
        #X_1 = self.GCN(x, edge_index) #
        #M1_readout = gap(X_1, batch)
        M1_con = self.projection_head_message(M1_P)
        
        # localpooling 1 
        X_L1, edge_index_local1, edge_attr_local1, batch_l1, perm1_1, score = self.LP1(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)
        L1_readout = gap(X_L1, batch_l1)
        L1_con = self.projection_head_local(L1_readout)

        X_S1, edge_index_sem1, edge_attr_sem2, batch_s1, perm2_1, _ = self.SP1(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)
        S1_readout = gap(X_S1, batch_s1) 
        S1_con = self.projection_head_semantic(S1_readout)

        #产生第一层的对比损失
        self.L1M1_loss = global_global_loss_(M1_con, L1_con)
        self.S1M1_loss = global_global_loss_(M1_con, S1_con)
        self.L1S1_loss = global_global_negative_loss_(L1_con, S1_con)
        self.CON_LOSS1 = self.L1M1_loss + self.S1M1_loss + self.L1S1_loss

        #self.YH_loss1 = YHloss.compute(self, anchor = M1_con, sample1 = L1_con, sample2 = S1_con )
        self.YH_loss1 = PAP1Loss.compute(self, anchor = M1_con, sample1 = L1_con, sample2 = S1_con )

        z1 = M1_P#z1 = torch.cat((M1_P, (L1_readout + S1_readout)/2), 1)

        #fusion 产生下一层的输入
        x1, batch1, edge_index1, edge_attr1 = fuse_x(perm1_1, perm2_1, batch_l1, X_L1, X_S1, batch, edge_index, edge_attr)
        

        #d=第二层
        X_M2, M2_P = self.M2(x1, edge_index1, batch1)
        M2_readout = gap(X_M2,batch1) #
        M2_con = self.projection_head_message(M2_P)
        # localpooling 1 
        X_L2, edge_index_local, edge_attr_local, batch_l2, perm1_2, _ = self.LP2(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)
        L2_readout = gap(X_L2, batch_l2)
        L2_con = self.projection_head_local(L2_readout)
        X_S2, edge_index_local2, edge_attr_local2, batch_s2, perm2_2, _ = self.SP2(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)

        #X_S2, edge_index_local2, edge_attr_local2, batch_s2, perm2_2, _ = self.SP2(x, edge_index, edge_attr = None, batch = batch, attn = None)
        S2_readout = gap(X_S2, batch_s2) 
        S2_con = self.projection_head_semantic(S2_readout)

        self.L2M2_loss = global_global_loss_(M2_con, L2_con)
        self.S2M2_loss = global_global_loss_(M2_con, S2_con)
        self.L2S2_loss = global_global_negative_loss_(L2_con, S2_con)
        self.CON_LOSS2 = self.L2M2_loss + self.S2M2_loss + self.L2S2_loss
        self.M1M2_loss = global_global_loss_(M2_con, M1_con)
        #self.YH_loss2 = YHloss.compute(self, anchor = M2_con, sample1 = L2_con, sample2 = S2_con )
        self.YH_loss2 = PAP1Loss.compute(self, anchor = M2_con, sample1 = L2_con, sample2 = S2_con )
        #z2 = M2_readout + L2_readout + S2_readout
        z2 = M2_P#z2 = torch.cat((M2_P, (L2_readout + S2_readout)/2), 1)

        x2, batch2, edge_index2, edge_attr2 = fuse_x(perm1_2, perm2_2, batch_l2, X_L2, X_S2, batch, edge_index, edge_attr)


        #第三层
        X_M3, M3_P = self.M3(x2, edge_index2, batch2)
        M3_readout = gap(X_M3,batch2) #
        M3_con = self.projection_head_message(M3_P)
        # localpooling 1 
        X_L3, edge_index_local, edge_attr_local, batch_l3, perm1_3, _ = self.LP3(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)
        L3_readout = gap(X_L3, batch_l3)
        L3_con = self.projection_head_local(L3_readout)
        X_S3, edge_index_local3, edge_attr_local3, batch_s3, perm2_3, _ = self.SP3(x, edge_index, edge_attr = edge_attr, batch = batch, attn = None)
        S3_readout = gap(X_S3, batch_s3) 
        S3_con = self.projection_head_semantic(S3_readout)

        self.L3M3_loss = global_global_loss_(M3_con, L3_con)
        self.S3M3_loss = global_global_loss_(M3_con, S3_con)
        self.L3S3_loss = global_global_negative_loss_(L3_con, S3_con)
        self.CON_LOSS3 = self.L3M3_loss + self.S3M3_loss + self.L3S3_loss
        self.M2M3_loss = global_global_loss_(M2_con, M3_con)
        self.M3M1_loss = global_global_loss_(M3_con, M1_con)

        # local_global loss
        self.M1M2_gl_loss = local_global_loss_(X_M1, M2_P, batch)
        self.M2M3_gl_loss = local_global_loss_(X_M2, M3_P, batch1)
        self.M3M1_gl_loss = local_global_loss_(X_M3, M1_P, batch2)
        #self.YH_loss3 = YHloss.compute(self, anchor = M3_con, sample1 = L3_con, sample2 = S3_con )
        self.YH_loss3 = PAP1Loss.compute(self, anchor = M3_con, sample1 = L3_con, sample2 = S3_con )

        #self.H_loss = YHloss.compute(self, anchor = M1_con, sample1 = M2_con, sample2 = M3_con  )
        self.H_loss = PAP1Loss.compute(self, anchor = M1_con, sample1 = M2_con, sample2 = M3_con  )
        #z3 = M3_readout + L3_readout + S3_readout
        z3 = M3_P#z3 = torch.cat((M3_P, (L3_readout + S3_readout) / 2), 1)
        #fusion出新的图 
        #x3, batch3, edge_index3, edge_attr3 = fuse_x(perm1_3, perm2_3, batch_l3, X_L3, X_S3, batch, edge_index, edge_attr)

        #最后合并出最终的 表征
        x = (z1 + z2 + z3) / 3
        return x 

    def compute_conloss(self):
        return (self.w1 * self.CON_LOSS1 + self.w2 * self.CON_LOSS2 + self.w3 * self.CON_LOSS3 )
        #self.w1, self.w2, self.w3 = 0.8, 0.2, 0.1
    def hierarchical_loss(self):
        return (self.w6 * self.M1M2_loss + self.w7 * self.M2M3_loss  + self.w8 * self.M3M1_loss )
        # 1 1 0.5

    def compute_gl_loss(self):

        return (self.M1M2_gl_loss + self.M2M3_gl_loss + self.M3M1_gl_loss)/3 #(self.M1M2_gl_loss + self.M2M3_gl_loss + self.M3M1_gl_loss)
    
    def compute_YH_loss(self):
        return (self.YH_loss1+self.YH_loss2+self.YH_loss3)/3
    
    def compute_H(self):
        return self.H_loss


