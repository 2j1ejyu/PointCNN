import os
import argparse
import random

import torch
from dataloader import get_dataloader
from models import *
from augmentor import PointAugment_simple
from run import run

import pdb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():

    parser = argparse.ArgumentParser(description="Jaewon's PointCloud")
    #parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--dataset', type=str, default='ModelNet10')
    parser.add_argument('--val_split', type=str, default='False') # Split 1/5 of train data if True
    parser.add_argument('--model', type=str, default='PointCNN')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_epoch', type=int, default=200000)

    parser.add_argument('--use_wandb', type=str, default='False')
    parser.add_argument('--project_title', type=str, default='no_title')

    parser.add_argument('--model_save', type=str, default='False')
    parser.add_argument('--use_augmentation', type=str, default='True', help='conventional augmentation') # False: vanilla
    # augment parameters
    parser.add_argument('--rotate_sigma', type=int, default=180)  # degree range for uniform distribution, sigma for gaussian distribution (0~180)
    parser.add_argument('--rotate_axis', type=str, default='z')
    parser.add_argument('--rotate_gaussian', type=str, default='False')
    parser.add_argument('--jitter_sigma', type=float, default=0.01)
    parser.add_argument('--jitter_clip', type=float, default=0.03)
    parser.add_argument('--shift_range', type=float, default=0.1)
    parser.add_argument('--scale_low', type=float, default=0.8)
    parser.add_argument('--scale_high', type=float, default=1.25)
    parser.add_argument('--shuffle', type=str, default = 'True')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = int(args.dataset[-2:])
    train_loader, test_loader = get_dataloader(args, device)

    if args.model == 'PointCNN':
        model = PointCNN(num_classes).to(device)

    augmentor = PointAugment_simple(args)
    
    run(args, num_classes, train_loader, test_loader, model, augmentor, device)

if __name__ == "__main__":
    main()