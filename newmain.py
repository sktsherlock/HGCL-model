import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import  tqdm
import json
# from models.HGCL3 import HGCL3
from models.HGCL2 import HGCL2
from models.HGCL1 import HGCL1
import build_data
import random
from evalute_embedding import svc, linearsvc
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--HGCL_layer', type=int, default=2, help='model name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--far', type=int, default=3, help='FARPool training epoch')
parser.add_argument('--dataset', type=str, default='MUTAG', help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
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


def con_train(args, model, dataloader ):
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
            f.write('dataset{},batch{},seed{},trade_off{},loss{}\n'.format(args.dataset, args.batch_size, args.seed,  args.trade_off, s))
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


def main():
    #检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:1"
    else:
        args.device = "cpu"
    #设置随机种子
    set_seed(args.seed)
    #输出HGCL模型中的多种超参数
    print(f'HGCL Layer:{args.HGCL_layer}, FARPooling epoch:{args.far}, Pooling_ratio:{args.pooling_ratio}, Trade-off:{args.trade_off}')
    # 输出训练的数据，batchsize epoch 等
    print(f'Dataset: {args.dataset}, Epochs:{args.epochs}, Batch: {args.batch_size}, Seed: {args.seed}, Learning rate: {args.lr}')
    #加载数据
    loader = build_data.build_loader(args)  
    # 加载模型
    if args.HGCL_layer == 1:
        from models.HGCL1 import Config
        config = Config()
        model = HGCL1(config, args, 2, 32).to(args.device)
    
    elif args.HGCL_layer == 2:
        from models.HGCL2 import Config
        config = Config()
        model = HGCL2(config, args, 2, 32).to(args.device)
        
    elif args.HGCL_layer == 3:
        from models.HGCL3 import Config
        config = Config()
        model = HGCL3(config, args, 2, 32).to(args.device)
    #第一阶段 训练FARPool部分，避免对比学习任务过于简单
    # train_pool(args, model, loader)
    #第二阶段 对比学习部分
    con_train(args, model, loader)
    #训练与测试完毕
    print('End')
if __name__ == '__main__':
    main()
