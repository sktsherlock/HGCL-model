import os
import torch
import argparse
import warnings
import numpy as np
from utils import get_split, SVMEvaluator
from tqdm import trange, tqdm
import json
from models.YHNet import YHNet, Config 
from models.HGCL2 import HGCL2
from models.HGCL1 import HGCL1
import build_data
import random
from evalute_embedding import svc, linearsvc
torch.set_printoptions(threshold=np.inf)


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--HGCL_layer', type=int, default=3, help='model name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--far', type=int, default=3, help='FARPool training epoch')
parser.add_argument('--dataset', type=str, default='PTC_MR', help='DD/NCI1/NCI109/MUTAG/PROTEINS')
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--pooling_ratio', type=float, default=0.9, help='TopK pooling ration')
parser.add_argument('--trade_off', type=float, default=0.5, help='trade-off: 0, 0.25, 0.5, 0.75, 1')
args = parser.parse_args()


# def test1(args, encoder_model, dataloader):
#     encoder_model.eval()
#     x = []
#     y = []
#     for data in dataloader:
#         data = data.to(args.device)
#         if data.x is None:
#             num_nodes = data.batch.size(0)
#             data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
#         g, _ = encoder_model(data.x, data.edge_index, data.batch)
#         x.append(g)
#         y.append(data.y)

#     x = torch.cat(x, dim=0)
#     y = torch.cat(y, dim=0)
#     split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
#     result = SVMEvaluator(linear=True)(x, y, split)
#     return result["accuracy"], result["std"]


def test2(args, encoder_model, dataloader):
    encoder_model.eval()
    x, y = encoder_model.conv1.get_embeddings(args.device, dataloader)
    acc_mean, acc_std = svc(x, y)
    print('Before training: acc_mean = ', acc_mean, '  acc_std = ', acc_std)
    with open('randomDD2_result.txt', 'a') as f:
        f.write(str(acc_mean) + ' ')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def con_train(args, epochs, dataset, seed, layers, batch, model, dataloader, mode):
    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)  # 仿照GraphCL里面的设置
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_interval = 5
    pbar = tqdm(range(1, epochs + 1), ncols=100)
    for epoch in pbar:
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _ = model(data.x, data.edge_index, data.batch, args.device)
            if mode == '0':
                loss = model.compute_inner()
            elif mode == '0.25':
                loss = 0.75 * model.compute_inner() + 0.25 * model.compute_hier()
            elif mode == '0.5':
                loss = (model.compute_inner() + model.compute_hier())/2
            elif mode == '0.75':
                loss = 0.25 * model.compute_inner() + 0.75 * model.compute_hier() 
            elif mode == '1':
                loss = model.compute_hier()
            # 返回对比损失
            dis_loss = loss
            dis_loss.backward()

            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        with open('526mutag.log', 'a') as f:
            s = json.dumps(loss_all / len(dataloader))
            f.write('dataset{},batch{},seed{},mode{},loss{}\n'.format(dataset, batch, seed,  mode, s))
        if epoch % log_interval == 0:
            model.eval()
            x, y = model.get_embeddings(args.device, dataloader)
            # x = []
            # y = []
            # for data in dataloader:
            #     data = data.to(args.device)
            #     g, _ = model(data.x, data.edge_index, data.batch, args.device)
            #     x.append(g)
            #     y.append(data.y)

            # x = torch.cat(x, dim=0)
            # y = torch.cat(y, dim=0)
            acc_mean, acc_std = svc(x, y)
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)

            with open('526mutag.txt', 'a') as f:
                f.write(str(acc_mean) + ' ')

    
    with open('526mutag.txt', 'a') as f:
        f.write('\n')


def con_train_topk(args, epochs, dataset, seed, layers, batch, model, dataloader):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _ = model(data.x, data.edge_index, data.batch, args.device)

            loss = model.compute_YH_loss()
            # 返回对比损失
            dis_loss = loss  # 10 * h_loss
            dis_loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))


def train_pool(args, epochs, dataset, seed, layers, batch, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr =0.01)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _ = model(data.x, data.edge_index, data.batch, args.device)
            loss = model.compute_Farloss()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))



def main():
    seeds = [0, 1, 2, 3, 4]
    batch = [32, 64]
    epoch = [20]
    layers = [2]
    modes = ['0', '0.25', '0.5', '0.75', '1']
    ds = ['MUTAG']   #['MUTAG','PROTEINS','IMDB-BINARY','PTC_MR','DD','NCI1','COLLAB','REDDIT-BINARY','REDDIT-MULTI-5K'] #['REDDIT-MULTI-5K','REDDIT-BINARY']#['PROTEINS','COLLAB','NCI1','REDDIT-BINARY','DD']#['IMDB-BINARY','PTC_MR','PROTEINS','IMDB-MULTI','MUTAG']#['MUTAG', 'NCI1', 'PTC_MR', 'IMDB-MULTI','IMDB-BINARY','PROTEINS', 'COLLAB', 'DD','REDDIT-BINARY']#['MUTAG','IMDB-MULTI','MUTAG','IMDB-BINARY','NCI1','PROTEINS','PTC_MR','COLLAB','DD','REDDIT-BINARY'] #,'IMDB-BINARY', 'COLLAB']# 'PTC_MR', 'IMDB-BINARY', 'DD', 'REDDIT-BINARY', 'IMDB-MULTI', ]  # ['COLLAB', 'IMDB-BINARY', 'MUTAG','IMDB-MULTI', 'PROTEINS', 'COLLAB']
    for d in ds:
        print(f'-------------------------------{d}------------------------------')
        for b in batch:
            for m in modes:
                for e in epoch:
                    for i in range(5):
                        if torch.cuda.is_available():
                            args.device = "cuda:0"
                        else:
                            args.device = "cpu"

                        seed = seeds[i]
                        set_seed(seed)
                        #HGCL_layer = args.HGCL_layer
                        print(f'Dataset: {d}, Layer:{args.HGCL_layer}, Batch: {b}, Epoch: {e}, seed: {seed}, far:{f}, mode:{m} ')
                        loader, loader_test, loader_draw = build_data.build_loader(args, b, d)  # 后面改
                        config = Config()
                        if args.HGCL_layer == 1:
                            model = HGCL1(config, args, 2, 32).to(args.device)
                        elif args.HGCL_layer == 2:
                            model = HGCL2(config, args, 2, 32).to(args.device)
                        elif args.HGCL_layer == 3:
                            model = YHNet(config, args, 2, 32).to(args.device)
                        # elif l == 4:
                        #     model = HGCL4(config, args, l, 32).to(args.device)
                        #model = YHNet(config, args, l, 32).to(args.device)
                        #model = HGCL1(config, args, l, 32).to(args.device)
        
                        train_pool(args, f, d, seed, 2, b, model, loader)
                        con_train(args, e, d, seed, 2, b, model, loader, m)

                        
                        if i == 4:
                            with open('526mutag.txt', 'a') as f:
                                f.write('\n')
                        pass 


if __name__ == '__main__':
    main()
