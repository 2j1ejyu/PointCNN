from abc import ABC, abstractmethod
import torch
from torch_geometric.nn import knn


class BaseNeighbourFinder(ABC):
    def __call__(self, x, y, batch_x, batch_y):
        return self.find_neighbours(x, y, batch_x, batch_y)

    @abstractmethod
    def find_neighbours(self, x, y, batch_x, batch_y):
        pass

    def __repr__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__)


class KNNNeighbourFinder(BaseNeighbourFinder):
    def __init__(self, k):
        self.k = k

    def find_neighbours(self, x, y, batch_x, batch_y):
        return knn(x, y, self.k, batch_x, batch_y)


class DilatedKNNNeighbourFinder(BaseNeighbourFinder):
    # return k neighbors for each y
    def __init__(self, k, dilation):
        self.k = k
        self.dilation = dilation
        self.initialFinder = KNNNeighbourFinder(k * dilation)

    def find_neighbours(self, x, y, batch_x, batch_y):
        # find the self.k * self.dilation closest neighbors in x for each y
        center_idx, neighbor_idx = self.initialFinder.find_neighbours(x, y, batch_x, batch_y)

        # pick the neighbors from them
        index = torch.arange(0, center_idx.shape[0], step=self.dilation, dtype=torch.long, device=center_idx.device)
        center_idx, neighbor_idx = center_idx[index], neighbor_idx[index]

        return center_idx, neighbor_idx

