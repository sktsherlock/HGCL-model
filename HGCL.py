import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import json
from build_data import *
import random
import wandb
import GCL.augmentors as A
from eval import *
from models.Encoder import GConv, HDGCL
from torch_geometric.data import DataLoader
from GCL.models import DualBranchContrast
import GCL.losses as L
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling, EdgePooling

# ! wandb
WANDB_API_KEY = '9e4f340d3a081dd1d047686edb29d362c8933632'
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--hidden', type=int, default=128, help='hidden layers')
parser.add_argument('--layers', type=int, default=2, help='layers')
parser.add_argument('--tradeoff', type=float, default=0.8, help='权重')
parser.add_argument('--inner', type=float, default=0.8, help='权重')
parser.add_argument('--mix', type=float, default=0.8, help='权重')
parser.add_argument('--mixup', type=float, default=0.8, help='权重')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--min_score', type=float, default=None, help='pooling min_score')
parser.add_argument('--edge_drop', type=float, default=0.2, help='pooling ratio')
parser.add_argument('--feature_mask', type=float, default=0.2, help='pooling ratio')
parser.add_argument('--up', type=float, default=0.4, help='the threshold of the tradeoff')
parser.add_argument('--eval_patience', type=int, default=10, help='the patience of evaluate')
parser.add_argument('--num_runs', type=int, default=5, help='the patience of evaluate')
parser.add_argument('--warmup_epochs', type=int, default=100, help='the number of warmup_epochs')
parser.add_argument('--test_init', type=bool, default=False, help='whether test the initial state')
parser.add_argument('--add_to_edge_score', type=float, default=0.5, help='add_to_edge_score')
parser.add_argument('--pooling', type=str, default='topk', help='Different pooling methods')
parser.add_argument('--augment', type=str, default='FE', help='Select Augment Way')

args = parser.parse_args()
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
wandb.config = args
wandb.init(project="HGCL", entity="sher-hao", config=args, reinit=True)


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def add_noise(x, tradeoff):
    if tradeoff > 0:
        if tradeoff <= 1:
            x = (1 - tradeoff) * x + torch.randn_like(x) * tradeoff
        else:
            raise ValueError('tradeoff <= 1')
    return x


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    epoch_loss_0 = 0
    epoch_loss_1 = 0
    log_interval = 10
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        g, g3, g6, g1, g2, g4, g5, g7, g8 = encoder_model(data.x, data.edge_index, data.batch)
        # Inner Contrasting and Hierarchical Contrasting
        #! Mapping to the Contrastive Space
        g, g1, g2 = [encoder_model.graph_encoder.project(g) for g in [g, g1, g2]]
        g3, g4, g5 = [encoder_model.sub_encoder1.project(g) for g in [g3, g4, g5]]
        g6, g7, g8 = [encoder_model.sub_encoder2.project(g) for g in [g6, g7, g8]]
        #! Contrasting Loss
        loss_inner = contrast_model(g1=g1, g2=g2) + contrast_model(g1=g4, g2=g5) + contrast_model(g1=g7, g2=g8)
        loss_hierarchical = contrast_model(g1=g, g2=g3) + contrast_model(g1=g3, g2=g6) + contrast_model(g1=g, g2=g6)

        loss = loss_inner + args.tradeoff * loss_hierarchical
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_0 += loss_inner.item()
        epoch_loss_1 += loss_hierarchical.item()

    return epoch_loss, epoch_loss_0, epoch_loss_1


def test(encoder_model, dataloader):
    encoder_model.eval()
    x0 = []
    x1 = []
    x2 = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g, g1, g2 = encoder_model.get_embedding(data.x, data.edge_index, data.batch)
        # final_0 = g
        final_1 = args.mixup* g + (1 - args.mixup) * (g1 + g2) / 2

        # x0.append(final_0)
        x1.append(final_1)
        y.append(data.y)
    # x_0 = torch.cat(x0, dim=0)
    x_1 = torch.cat(x1, dim=0)
    y = torch.cat(y, dim=0)

    # graph_acc_mean, acc_std = svc(x_0, y)
    mix_acc_mean, acc_std = svc(x_1, y)

    return mix_acc_mean #graph_acc_mean


def diffusion(args, epoch, way='stage'):
    if way == 'stage':
        if epoch <= 10:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.01, 0.05).float()
        elif epoch <= 20:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.1, 0.2).float()
        else:
            tradeoff = args.up
    else:
        raise ValueError('the other way is not implement')
    return tradeoff


def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    # G_Acc_Mean = []
    M_Acc_Mean = []
    for i in range(args.num_runs):
        set_seed(i)

        dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        input_dim = max(dataset.num_features, 1)

        # test the randomization model
        print('--------------------------------')
        print('Augmentation')
        if args.augment in {'Random'}:
            aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
            aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
        elif args.augment in {'FE'}:
            aug1 = A.Compose([A.FeatureMasking(pf=args.feature_mask)])
            aug2 = A.Compose([A.EdgeRemoving(pe=args.edge_drop)])
        elif args.augment in {'SNE'}:
            aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
            aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
        else:
            raise ValueError('Not implement for the Augmented way!')
        print('Model Initationize')
        gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=args.layers).to(args.device)
        gconv1 = GConv(input_dim=args.hidden * args.layers, hidden_dim=args.hidden, num_layers=args.layers).to(args.device)
        gconv2 = GConv(input_dim=args.hidden * args.layers, hidden_dim=args.hidden, num_layers=args.layers).to(
            args.device)

        if args.pooling in {'topk', 'TopK'}:
            pool_1 = TopKPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
            pool_2 = TopKPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
        elif args.pooling in {'SAGPooling', 'SAG'}:
            pool_1 = SAGPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
            pool_2 = SAGPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
        else:
            raise ValueError('Not implement')
        encoder_model = HDGCL(graph_encoder=gconv, augmentor=(aug1, aug2), pool_1=pool_1, pool_2=pool_2, sub_encoder1=gconv1, sub_encoder2=gconv2).to(args.device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(args.device)
        optimizer = Adam(encoder_model.parameters(), lr=args.lr)

        log_interval = args.eval_patience
        # G_Accuracy = []
        M_Accuracy = []
        with tqdm(total=args.epochs, desc=f'(T){i}') as pbar:
            for epoch in range(1, args.epochs + 1):
                loss, loss_0, loss_1 = train(encoder_model, contrast_model, dataloader, optimizer)
                if epoch % log_interval == 0:
                    encoder_model.eval()
                    mix_acc_mean = test(encoder_model, dataloader)
                    # G_Accuracy.append(graph_acc_mean)
                    M_Accuracy.append(mix_acc_mean)
                    # wandb.log({'G_Accuracy': graph_acc_mean, 'M_Accuracy': mix_acc_mean})
                    wandb.log({'M_Accuracy': mix_acc_mean})

                pbar.set_postfix({'loss': loss})
                wandb.log({"loss": loss, "Loss_inner": loss_0, "Hierarchical_loss": loss_1})
                pbar.update()

        wandb.log({'M_Acc': mix_acc_mean})

        wandb.log({'Best M_Acc': max(M_Accuracy)})
        M_Acc_Mean.append(max(M_Accuracy))

    # print('Run 5, the mean accuracy is {}'.format(np.mean(Acc_Mean)))
    wandb.log({'Mean M_Acc': np.mean(M_Acc_Mean), 'M_Std': np.std(M_Acc_Mean) })


def test_init():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    Acc = []
    with tqdm(total=5, desc='(T)') as pbar:
        for i in range(5):
            set_seed(i)

            dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
            input_dim = max(dataset.num_features, 1)

            # test the randomization model
            # aug1 = A.Identity()
            # aug2 = A.Identity()
            #
            # gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=2).to(args.device)
            # encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(args.device)
            # optimizer = Adam(encoder_model.parameters(), lr=0.01)
            # initial_acc_mean = test(encoder_model, dataloader)
            initial_acc_mean = test_0(dataloader)
            Acc.append(initial_acc_mean)
            pbar.set_postfix({'Acc': initial_acc_mean})
            pbar.update()
        wandb.log({"Accuracy": np.mean(Acc), "Std": np.std(Acc)})


if __name__ == '__main__':
    if args.test_init == False:
        main()
    else:
        test_init()
