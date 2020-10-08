from math import ceil
import torch
from torch.nn import Sequential as S, Linear as L, BatchNorm1d as BN
from torch.nn import ELU, Conv1d
from torch_geometric.data import Batch
from neighbour_finder import DilatedKNNNeighbourFinder
from sampling import RandomSampler, FPSSampler

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        """"""
        x = x.view(*self.shape)
        return x

class XConv(torch.nn.Module):
    """
    C_in: # of in channel
    C_out: # of out channel
    dim: dimension
    K: # of neighboring points (kernel size)
    C_delta: size of the output of MLP_delta (changed dimension)
    """

    def __init__(self, C_in, C_out, dim, K):
        super(XConv, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.dim = dim
        C_delta = C_out // 2 if C_in == 0 else C_in // 4
        self.C_delta = C_delta

        # mlp_delta (pointwise MLP)
        self.mlp_delta = S(        #Input: (N*K,dim) Output: (N*K,C_delta)
            L(dim, C_delta), #FC
            ELU(), 
            BN(C_delta), 
            L(C_delta, C_delta), #FC
            ELU(), 
            BN(C_delta)
        )

        # mlp that learns the X-transformation
        self.mlp_X = S(                              #Input: (N,dim*K) Output: (N,K,K)
            L(dim*K, K**2),                          #out: (N,K**2)
            ELU(),
            BN(K**2),
            Reshape(-1, K, K),                       #out: (N, K, K)
            Conv1d(K, K ** 2, K, groups=K),  #depthwise convolution   #out: (N, K**2, 1)
            ELU(),
            BN(K ** 2),
            Reshape(-1, K, K),                       #out: (N, K, K)
            Conv1d(K, K ** 2, K, groups=K),          #out: (N, K**2, 1)
            BN(K ** 2),
            Reshape(-1, K, K)                        #out: (N, K, K)
        )
  
        DM = int(ceil(C_out/(C_in + C_delta)))  #depth multiplier
        C_in += C_delta

        self.Conv = S(    # input: (N, C_in + C_delta, K), output: (N, C_out)
            Conv1d(C_in, C_in * DM, K, groups=C_in),  # out: (N, (C_in + C_delta)*DM, 1)
            Reshape(-1, C_in * DM),                   # out: (N, (C_in + C_delta)*DM)
            L(C_in * DM, C_out),                      # out: (N, C_out)
        )
        
    def forward(self, x, pos, idx, neighbor_idx): # N1: B*inN, N2: B*outN
        # x: (N1,C), pos: (N1,dim), idx: (N2,1), neighbor_idx: (N2,K)
        K = neighbor_idx.shape[1]
        
        relPos = pos[neighbor_idx]-pos[idx]   # (N2,K,dim) relative positions
        x_star = self.mlp_delta(relPos.view(-1,self.dim)).view(-1, K, self.C_delta) 
        # (N2, K, C_delta)

        if x is not None:
            x = x[neighbor_idx]           # (N2,K,C)
            x_star = torch.cat([x_star, x], dim=-1)   # (N2, K, C_delta + C)

        transform_matrix = self.mlp_X(relPos.view(-1, K*self.dim))  # (N2,K,K)

        x_transformed = torch.bmm(transform_matrix, x_star)  # (N2,K, C_delta + C)
        x_transformed = x_transformed.transpose(1,2) # (N2,C_delta + C, K)

        out = self.Conv(x_transformed)    # (N2, C_out)

        return out


class PointCNNConvDown(torch.nn.Module):
    def __init__(self, outN, K, D, C_in, C_out):
        super(PointCNNConvDown, self).__init__()
        self.K = K
        self.outN = outN
        self.sampler = RandomSampler(num_to_sample=outN) 
        self.neighbour_finder = DilatedKNNNeighbourFinder(K, D) 
        self.Xconv = XConv(C_in, C_out, 3, K)

    def forward(self, data):
        data_out = Batch()
        # data.x: (B*inN,Cin), data.pos: (B*inN,3), data.batch: (B*inN)
        if data.batch.shape[0]/(data.batch[-1]+1) == self.outN:
            idx = torch.arange(data.batch.shape[0], device = data.pos.device)
        else:
            idx = self.sampler(data.pos, batch=data.batch)   #index of representative points: (B*outN)
        # data_out.idx = idx
        data_out.pos = data.pos[idx]  # (B*outN,3)
        data_out.batch = data.batch[idx]  # (B*outN)
        _, neighbor_idx = self.neighbour_finder(data.pos, data.pos[idx], batch_x=data.batch, batch_y=data.batch[idx])
        neighbor_idx = neighbor_idx.view(-1, self.K) #(B*outN*K) -> (B*outN,K)
        idx = idx.unsqueeze(-1)   #(B*outN) -> (B*outN,1)

        data_out.x = self.Xconv(data.x, data.pos, idx, neighbor_idx)  # (B*outN, C_out)

        """
        for key in data.keys:
            if key not in data_out.keys:
                setattr(data_out, key, getattr(data, key, None))
        """
        return data_out    

    