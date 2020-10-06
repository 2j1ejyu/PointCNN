import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from PointCNN_modules import PointCNNConvDown

from torch.nn import ELU, Dropout, Sequential as S, Linear as L, BatchNorm1d as BN


class PointCNN(nn.Module):
    def __init__(self, num_classes):
        super(PointCNN, self).__init__()

        self.num_classes = num_classes
        self.xconvs = S(
            PointCNNConvDown(1024, 8, 1, 0, 48),   # (outN, K, D, C_in, C_out)
            PointCNNConvDown(384, 12, 2, 48, 96),
            PointCNNConvDown(128, 16, 2, 96, 192),
            PointCNNConvDown(128, 16, 3, 192, 384)
        )

        self.fc = S(
            L(384, 256),
            ELU(),
            BN(256),
            L(256,128),
            ELU(),
            BN(128),
            Dropout(0.5),
            L(128,num_classes)
        )
    
    def forward(self, data, *args, **kwargs): 
        out = self.xconvs(data)
        out = out.x                 # out.x: (B*outN, C_out)=(B*128, 384)
        out = self.fc(out)          # out: (B*128, num_classes)
        out = torch.mean(out.view(-1,128,self.num_classes), dim=1) # (B,num_classes)
        return F.log_softmax(out, dim=-1)

