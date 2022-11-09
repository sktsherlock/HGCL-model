import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import json
import build_data
import random
from evalute_embedding import svc, linearsvc
from models.HNet import HNet, Config, innercl
from models.Encoder import HGCL

torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--HGCL_layer', type=int, default=3, help='model name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# add num_layers in parser
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--far', type=int, default=3, help='FARPool training epoch')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--pooling_ratio', type=float, default=0.9, help='TopK pooling ration')
parser.add_argument('--trade_off', type=float, default=0.5, help='trade-off: 0, 0.25, 0.5, 0.75, 1')
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def con_train(args, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_interval = 5
    pbar = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in pbar:
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _ = model(data.x, data.edge_index, data.batch, args.device)
            loss = (1 - args.trade_off) * model.compute_inner() + args.trade_off * model.compute_hier()
            # 返回对比损失
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        with open('CLtraining.log', 'a') as f:
            s = json.dumps(loss_all / len(dataloader))
            f.write('dataset{},batch{},seed{},trade_off{},loss{}\n'.format(args.dataset, args.batch_size, args.seed,
                                                                           args.trade_off, s))
        if epoch % log_interval == 0:
            model.eval()
            x, y = model.get_embeddings(args.device, dataloader)
            acc_mean, acc_std = svc(x, y)
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)
            with open('CLtraining.txt', 'a') as f:
                f.write(str(acc_mean) + ' ')
    with open('CLtraining.txt', 'a') as f:
        f.write('\n')


def train_pool(args, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, int(args.far) + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _ = model(data.x, data.edge_index, data.batch, args.device)
            loss = model.compute_Farloss()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))


def train(cf, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr)
    log_interval = 5
    pbar = tqdm(range(1, cf.epochs + 1), ncols=100)
    for epoch in pbar:
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(cf.device)


            if data.x is None:
                data.x = torch.ones(data.num_nodes, 1).to(cf.device)
            _, g1, g2, g3, p1, p2, p3 = model(data)

            loss = (innercl(p1, p2) + innercl(p2, p3) + innercl(p1, p3))/3 + (innercl(g1, g2) + innercl(g2, g3) + innercl(g1, g3))/3
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        with open('CLtraining.log', 'a') as f:
            s = json.dumps(loss_all / len(dataloader))
            f.write('dataset{},batch{},seed{},trade_off{},loss{}\n'.format(cf.dataset, cf.batch_size, cf.seed,
                                                                           cf.trade_off, s))
        if epoch % log_interval == 0:
            model.eval()
            x, y = model.get_embeddings(cf.device, dataloader)
            acc_mean, acc_std = svc(x, y)
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)


def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    # 设置随机种子
    set_seed(args.seed)
    # 加载数据
    loader = build_data.build_loader(args)
    # 加载模型
    model = HNet(args=args, config=Config())
    model.to(args.device)
    # 训练模型
    train(args, model, loader)


if __name__ == '__main__':
    main()
