import os.path as osp
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

def build_loader(args):
    dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
    args.num_classes = dataset.num_classes
    loader = DataLoader(dataset, batch_size = args.batch_size,shuffle = True,  num_workers = 20)
    args.num_features = max(dataset.num_features, 1)

    return loader 
