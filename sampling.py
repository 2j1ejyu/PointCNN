import math
import torch
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """ 
        If num_to_sample is provided, sample exactly num_to_sample points. 
        Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None):
        # num_to_sample not implemented yet
        if (num_to_sample is not None) and (ratio is not None):
             raise ValueError("Can only specify ratio or num_to_sample, not several !")

        if num_to_sample is not None:
            self.num_to_sample = num_to_sample

        elif ratio is not None:
            self.ratio = ratio

        else:
            raise Exception('At least ["ratio, num_to_sample"] should be defined')

    def get_num_to_sample(self, point_num) -> int:
        if hasattr(self, "num_to_sample"):
            return self.num_to_sample
        else:
            return math.floor(point_num * self.ratio)

    def get_ratio_to_sample(self, point_num) -> float:
        if hasattr(self, "ratio"):
            return self.ratio
        else:
            return self.num_to_sample / float(point_num)

    def __call__(self, pos, x=None, batch=None):
        # pos: (B*N,3) , x: (B*N, C), batch: (B*N)
        # B: batch num, N: point num, C: feature num of x
        if(batch is None):
            point_num = pos[0]
        else:
            point_num = batch.shape[0] / (batch[-1]+1)
        return self.sample(pos, point_num, batch=batch, x=x)
    
    @abstractmethod
    def sample(self, pos, point_num, x=None, batch=None):
        pass

class FPSSampler(BaseSampler):
    def sample(self, pos, point_num, batch=None, **kwargs):
        from torch_geometric.nn import fps

        return fps(pos, batch=batch, ratio=self.get_ratio_to_sample(point_num)) 
        # out: (floor(ratio*pos[0]))


class RandomSampler(BaseSampler):
    def sample(self, pos, point_num, batch=None, **kwargs):
        batch_size = pos.shape[0] / point_num
        w = torch.ones((batch_size,point_num),device = pos.device)
        idx = torch.multinomial(w, self.get_num_to_sample(point_num)) + (point_num * torch.arange(batch_size,device = pos.device).unsqueeze(-1))
        idx = idx.view(-1)
        return idx

