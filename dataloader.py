from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from augmentor import PointAugment_simple
from torch_geometric.data import DataLoader
import math

def get_dataloader(args, device):
    path = args.dataset
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(args.num_points)
    
    train_dataset = ModelNet(
        'dataset/' + path,
        name=args.dataset[-2:],
        train=True,
        transform=transform,
        pre_transform=pre_transform)
    train_dataset.data.to(device)
    
    test_dataset = ModelNet(
        'dataset/' + path,
        name=args.dataset[-2:],
        train=False,
        transform=transform,
        pre_transform=pre_transform)
    test_dataset.data.to(device)
    
    if args.val_split=='True':
        split = math.floor(len(train_dataset)/5)
        train_loader = DataLoader(train_dataset[:-split], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset[-split:], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        return train_loader, test_loader