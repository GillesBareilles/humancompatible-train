import torch
from typing import Iterable, List, Callable
from itertools import cycle
import numpy as np

def _dataloader_from_subset(dataset, indices, *args, **kwargs):
    data_s = torch.utils.data.Subset(dataset, indices)
    loader_s = torch.utils.data.DataLoader(data_s, *args, **kwargs)
    return loader_s

class FairnessConstraint():
    
    def __init__(self,
            dataset: torch.utils.data.Dataset,
            group_indices: Iterable[Iterable[int]],
            fn: Callable,
            batch_size: int = None,
            use_dataloaders = True
        ):
        self.dataset = dataset
        self.group_sets = [torch.utils.data.Subset(dataset, idx) for idx in group_indices]
        self.fn = fn
        if not (batch_size is None):
            self.batch_size = batch_size
            if use_dataloaders:
                self.dataloaders = [cycle(_dataloader_from_subset(dataset, idx, batch_size=batch_size))
                                    for idx
                                    in group_indices]
        
    def eval(self, net, sample, **kwargs):
        return self.fn(net, sample, **kwargs)
    
    def sample_loader(self):
        return [next(l) for l in self.dataloaders]
    
    def sample_dataset(self, N, rng: np.random.Generator=None):
        if rng is None:
            rng = np.random.default_rng()
        # returns len(group) points if N > len(group)
        return [group[rng.choice(N) if N < len(group) else rng.choice(len(group))] for group in self.group_sets]