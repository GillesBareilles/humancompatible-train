import typing
import torch
import torch.utils
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from typing import Iterable, Callable
from itertools import cycle
import random

LOADER_SAMPLE_CODE = 'dlr'
INDEX_SAMPLE_CODE = 'ind'

class StochasticConstraint():
    '''
    A stochastic constraint class.

    Attributes:
    self.data
    '''
    def __init__(
        self,
        fun: Callable,
        data_source: Dataset | Iterable[Dataset] | DataLoader | Iterable[DataLoader],
        subgroup_indices: dict[int: Iterable[int]] = None,
        batch_size = 1):

        self.fun = fun

        self.__n_subgroups = None

        if subgroup_indices is None:
            self.data_source, self._data_cycle = self._datasource_from_dataloaders(data_source,
            batch_size)
            self.__which_sampling = LOADER_SAMPLE_CODE
            if isinstance(data_source, (list, tuple)):
                self.__n_subgroups = len(data_source)
            elif isinstance(data_source, DataLoader):
                self.__batch_size = data_source.batch_size
            else:
                self.__batch_size = batch_size
        elif isinstance(data_source, Dataset):
            self.data_source, self._data_cycle = self._datasource_from_indices(data_source, subgroup_indices, 
            batch_size)
            self.__which_sampling = INDEX_SAMPLE_CODE
            self.__n_subgroups = len(subgroup_indices)
        else:
            raise TypeError(
                f'When using subgroup_indices, data_source must be an instance of Dataset; got {type(data_source)}')
        
        self._last_sample = None
        # self.__batch_size = batch_size
    



    ########################
    # DATA PREPARATION F-S #
    ########################
    
    def _datasource_from_dataloaders(self,
        data: Dataset | DataLoader | Iterable[Dataset | DataLoader],
        batch_size=1
        )-> tuple[DataLoader, cycle]:

        """To be called by constructor, make a (list of) dataloader cycle(s)
        from a dataset/dataloader or a list of datasets/dataloaders

        For the case where 1. the constraint is evaluated without dividing into subgroups
        (e.g. FNR < eps) or 2. the constraint depends on subgroups (e.g. |loss(sg_1) - loss(sg_2)| < eps)
        and each subgroup is given as its own dataset/dataloader
        """
        if isinstance(data, Dataset):
            data_source = DataLoader(data, batch_size=batch_size)
            _data_cycle = cycle(data_source)

        elif isinstance(data, DataLoader):
            data_source = data
            _data_cycle = cycle(data_source)

        elif isinstance(data, Iterable) and isinstance(data[0], Dataset):
            data_source = [DataLoader(d, batch_size=batch_size) for d in data]
            _data_cycle = [cycle(dl) for dl in data_source]

        elif isinstance(data, Iterable) and isinstance(data[0], DataLoader):
            data_source = data
            _data_cycle = [cycle(dl) for dl in data_source]

        return data_source, _data_cycle


    def _datasource_from_indices(self,
        data: Dataset,
        subgroup_indices: dict[int],
        batch_size=1
        ) -> tuple[DataLoader, cycle]:
        
        """To be called by constructor, makes a (list of) dataloader cycle(s)
        from a dataset and a dict of subgroup indices
        """
        
        sampler = SubgroupBalancedBatchSampler(subgroup_indices, per_subgroup_size=batch_size)
        data_source = DataLoader(data, batch_sampler=sampler,
                                 collate_fn=lambda b: subgroup_collate_fn(b, batch_size=batch_size))
        _data_cycle = cycle(data_source)

        return data_source, _data_cycle
        

    ### SAMPLING

    def _sample_batch_from_dataloader(self) -> list[Tensor]:
        if isinstance(self._data_cycle, list):
            batches_by_group = [next(ds) for ds in self._data_cycle]
        else:
            batches_by_group = next(self._data_cycle)
        return batches_by_group

    def _sample_batch_from_indices(self) -> list[Tensor]:
        batches_by_group = next(self._data_cycle)
        return batches_by_group

    def _sample_batch(self) -> tuple[ list[Tensor] | list[list[Tensor]], int ]:
        sample_size = 0
        if self.__which_sampling == LOADER_SAMPLE_CODE:
            sample = self._sample_batch_from_dataloader()
        elif self.__which_sampling == INDEX_SAMPLE_CODE:
            sample = self._sample_batch_from_indices()
        
        if isinstance(sample[0], list):
            # sample: [group 1: [Tensor (X), Tensor (y)],  group 2: [Tensor (X), Tensor (y)], ...]
            sample_size = len(sample[0][0])
        elif isinstance(sample[0], Tensor):
            # sample: [Tensor (X), Tensor (y)]
            sample_size = len(sample[0])
        return sample, sample_size


    def sample(self, batch_len=None) -> list[Tensor] | list[list[Tensor]]:
        size = 0
        if not batch_len:
            batch_len = 1
        samples = []
        while size < batch_len:
            sample, sample_size = self._sample_batch()
            samples.append(sample)
            size += sample_size
        # "glue" the tensors together into a "batch"
        if isinstance(samples[0][0], Tensor):
            # no subgroup separation: each element of `samples` is a tuple [Tensor, Tensor]
            Xs = []
            ys = []
            for sample in samples:
                Xs.append(sample[0])
                ys.append(sample[1])
            return [torch.cat(Xs), torch.cat(ys)]
        else:
            # each element of `samples` is a list of (tuples [Tensor, Tensor]] for each subgroup)
            X_by_group = []
            y_by_group = []
            for group in range(self.__n_subgroups):
                X_by_group.append([])
                y_by_group.append([])
                for single_sample in samples:
                    X_by_group[group].append(single_sample[group][0])
                    y_by_group[group].append(single_sample[group][1])

            y_by_group = [torch.cat(y_gr) for y_gr in y_by_group]
            X_by_group = [torch.cat(x_gr) for x_gr in X_by_group]

            structured_sample = [[X_by_group[gr], y_by_group[gr]] for gr in range(self.__n_subgroups)]
            return structured_sample


    ################################
    # EVALUATE ON A SAMPLE AND NET #
    ################################

    def eval(self, net: torch.nn.Module, sample: list[Tensor] | list[list[Tensor]]):
        return self.fun(net, sample)


###########
# HELPERS #
###########

def subgroup_collate_fn(batch, batch_size) -> list[Tensor]:
    """
    Custom collate function to maintain subgroup separation
    batch: list of subgroup batches from the sampler
    batch_size: number of samples in each subgroup (equal)

    returns:
        List of collated samples for each subgroup: List[Tensor(batch_size, *sample_shape)]
    """
    # Reorganize into [[samples[subgroup]] for subgroup in subgroups]
    organized_batch = []
    for subgroup_idx in range(len(batch)//batch_size):
        subgroup_batch = batch[subgroup_idx*batch_size : (subgroup_idx+1)*batch_size]
        # default_collate returns [X (batch_size, *sample_size) , y (batch_size)]
        subgroup_batch_xy = torch.utils.data.default_collate(subgroup_batch)
        organized_batch.append(subgroup_batch_xy)
    
    return organized_batch



class SubgroupBalancedBatchSampler(BatchSampler):
    def __init__(self, subgroup_indices, per_subgroup_size, shuffle=True, repeat=True):
        """
        subgroup_indices: Dictionary {subgroup_id: list_of_indices}
        samples_per_subgroup: Number of samples per subgroup per batch
        shuffle: Whether to shuffle subgroup indices each epoch
        """
        self.subgroup_indices = subgroup_indices
        self.samples_per_subgroup = per_subgroup_size
        self.shuffle = shuffle
        self.n_subgroups = len(subgroup_indices)
        self.repeat = repeat
        
        # Calculate maximum possible batches per epoch (based on smallest subgroup)
        self.batches_per_epoch = min(
            len(indices) // per_subgroup_size
            for indices in subgroup_indices.values()
        )


    def __iter__(self):
        # Shuffle subgroup indices if required
        subgroup_iterators = {}
        for subgroup, indices in self.subgroup_indices.items():
            idx = list(indices).copy()
            if self.shuffle:
                random.shuffle(idx)
            subgroup_iterators[subgroup] = cycle(idx) if self.repeat else iter(idx) 

        # Generate batches for one epoch
        batch_num = 0
        while True:
            if not self.repeat and batch_num >= self.batches_per_epoch:
                raise StopIteration
            batch = []
            for subgroup in self.subgroup_indices:
                # Get next samples_per_subgroup indices from this subgroup
                batch.extend([
                    next(subgroup_iterators[subgroup])
                    for _ in range(self.samples_per_subgroup)
                ])

            batch_num +=1
            yield batch

    def __len__(self):
        return self.batches_per_epoch