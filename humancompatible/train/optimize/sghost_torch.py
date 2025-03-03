from typing import Iterable
import torch.utils.data.dataloader
import torch.optim as optim
import numpy as np
from ot.utils import list_to_array
from ot.backend import get_backend
import warnings
import argparse
import numpy as np
import time
import scipy as sp
from typing import Callable
from scipy.optimize import linprog
from qpsolvers import solve_qp
from ot.utils import unif, dist, list_to_array
import autoray as ar
from torch.utils.data import Sampler
from stochastic_constraint import StochasticConstraint
import timeit
from itertools import cycle

# needed for sampling in ghost
# init each iteration
# class CustomBatchSampler(Sampler):
#     def __init__(self, batches):
#         self.batches = batches
#     def __iter__(self):
#         for batch in self.batches:
#             yield batch
#     def __len__(self):
#         return len(self.batches)

def SampleFromDataset(dataset: torch.utils.data.Dataset, idx=None):
    if isinstance(dataset, torch.utils.data.IterableDataset):
        raise NotImplementedError()
    else:
        return dataset[idx]

def SampleFromDataLoader(dataloader, d_iterator, size):
    samples = []
    while len(samples < size):
        try:
            samples.append(next(d_iterator))
        except:
            d_iterator = iter(dataloader)
            return next(d_iterator), d_iterator
        

def net_grads_to_tensor(net) -> torch.Tensor:
        

class StochasticGhost:
    def __init__(self,
                 net: torch.nn.Module,
                 loss_fn: Callable,
                 m_det: int,
                 m_st: int,
                 loss_grad: Callable=None,
                 det_constr_grad: Callable=None,
                 st_constr_grad: Callable=None,
                 det_constraint_fn: Callable=None,
                #  st_constraint_fn: Callable=None,
                 st_constraint: StochasticConstraint = None,
                 qp_solver: str='osqp',
                 **kwargs):
        """
        Solves an inequality-constrained optimization problem using the Stochastic Ghost method:
        min f(x) s.t. c(x) <= 0

        Args:
            net (torch.nn.Module): the network to train.
            # params (iterable): iterable of parameters to optimize.
            loss_fn (Callable): the objective function.
            m_det (int): number of deterministic constraints.
            m_st (int): number of stochastic constraints.
            det_constraint_fn (Callable, optional): the deterministic constraint function, expected: f(net) -> Tensor((m,)); 
            expected to return a tensor of constraint values.
            st_constraint_fn (Callable, optional): the stochastic constraint function, expected:
            f(net, inputs, labels) -> Tensor((m,)); expected to return a tensor of constraint values.
            solver (str, optional): the method to solve quadratic subproblems; default is OSQP.
            **kwargs: arguments for the PyTorch optimizer.
        """
        # self._params = params
        self.net = net
        
        self._loss_fn = loss_fn
        self._det_constraint_fn = det_constraint_fn

        self.st_constraint = st_constraint

        self.m_det = m_det
        self.m_st = m_st

        if qp_solver is None:
            self._qp_solver = 'osqp'
        else:
            self._qp_solver = qp_solver

    def optimize(self,
                 data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                 maxiter: int=3,
                 epochs: int=3,
                 verbose: bool=True) -> None:
        """
        Perform optimization using the Augmented Lagrangian method.

        Iteratively minimize the augmented Lagrangian function with respect to 
        the neural network parameters while updating the Lagrange multipliers and augmentation term.
        The method updates the network parameters in place and records optimization history.

        Args:
            data (torch.utils.data.Dataset): Dataset providing input data and labels for training.
            con_data (torch.utils.data.Dataset): Dataset providing input data and labels
              for evaluating stochastic constraints.
            constr_sampling_interval (int | str, optional): number of algorithm iterations between
              resampling data for stochastic constraint evaluation. 
            epochs (int, optional): Number of epochs. Default is 3.
            maxiter (int, optional): Number of iterations for updating the Lagrange multipliers per epoch. Default is 3.
            verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

        Returns:
            None 
        """
        
        
        
        if isinstance(data, torch.utils.data.Dataset):
            obj_dataloader = torch.utils.data.DataLoader(data,batch_size=1)
            # obj_data_iter = obj_dataloader._get_iterator()
        elif isinstance(data, torch.utils.data.DataLoader):
            raise NotImplementedError()
            obj_dataloader = data
        else:
            raise TypeError(f'Only Dataset and DataLoader accepted, got {type(data)}')
        obj_data_iter = cycle(obj_dataloader)

        self.loss_val = 0
        self.history = {'L': [], 'loss': [], 'constr': []}
        

        for iteration in range(0, maxiter):
            iter_start = timeit.default_timer()

            if self.stepdec == 'dimin':
                gamma = self.gamma0/(iteration+1)**self.zeta
            if self.stepdec == 'constant':
                gamma = self.gamma0
            if self.stepdec == 'slowdimin':
                gamma = gamma*(1-self.zeta*gamma)
            if self.stepdec == 'stepwise':
                gamma = self.gamma0 / (10**(int(iteration*self.zeta)))

            Nsamp = np.random.geometric(p=self.geomp)
            while (2**(Nsamp+1)) > self.N:
                Nsamp = np.random.geometric(p=self.geomp)

            mbatches = [1, 2**Nsamp, 2**Nsamp, 2**(Nsamp+1)]
            dsols = np.zeros((4, self.n))
            rng = np.random.default_rng(seed=42)
            # for each subproblem:
            for subp_batch_size in mbatches:
                # take needed nr of samples from objective dataset
                if isinstance(data, torch.utils.data.Dataset):
                    # produces a tensor of shape (subp_batch_size, sample.shape)
                    idx = rng.choice(len(data), size=subp_batch_size)
                    obj_batch = SampleFromDataset(data, idx)
                
                else:
                    # TODO: take a batch from dataloader, keep taking until reach at least subp_batch_size
                    # if user has batch size more than minimal required for stghost, not our problem
                    raise NotImplementedError
                    
                #     obj_batch = next(obj_data_iter)
                #     obj_batch, new_dl = SampleFromDataLoader(obj_dataloader, obj_data_iter, subp_batch_size)
                #     if not(new_dl is None):
                #         obj_data_iter = new_dl
                
                # calculate autograd jacobian of obj fun w.r.t. params
                self.net.zero_grad()
                outs = self.net(obj_batch[0])
                feval = self._loss_fn(outs, obj_batch[1])

                feval.backward()
                # TODO: store all param grads in tensor
                dfdw = []
                
                # take needed nr of samples from constraint dataset(s)
                # samples = [SampleFromDataLoader(dl, di, subp_batch_size) for dl, di in zip(con_dataloader, con_data_iter)]
                
                st_constraint_sample = self.st_constraint.sample(batch_len=subp_batch_size)
                
                # calculate autograd jacobian of constraints fun w.r.t. params
                self.net.zero_grad()
                st_constraint_eval = self.eval_constraints(st_constraint_sample)
                st_constraint_eval.backward()
                # TODO: store all param grads in tensor
                dcdw = []

                # use functional jac to get grads
                # solve subproblem
            
            
            
            # sample_idx_f = [rng.choice(np.arange(len(data)), size=s) for s in mbatches]
            # sample_idx_c = [rng.choice(np.arange(len(con_data)), size=s) for s in mbatches]
            # sampler_f = CustomBatchSampler(sample_idx_f)
            # sampler_c = CustomBatchSampler(sample_idx_c)

        #######################
        # stopping condition? #
        #######################

    def eval_constraints(self, st_sample):
        constraint_eval = []
        if self.m_det > 0:
            det_constraint_eval = self._det_constraint_fn(self.net)
            if det_constraint_eval.ndim == 0:
                det_constraint_eval = det_constraint_eval.reshape(1)
            constraint_eval.append(det_constraint_eval)
        if self.m_st > 0:
            # TODO: collect evals of each constraint into a tensor, then do max
            st_constraint_eval = self._st_constraint_fn(self.net, st_sample)
            st_constraint_eval = torch.max(st_constraint_eval, torch.zeros(1))
            if st_constraint_eval.ndim == 0:
                st_constraint_eval = st_constraint_eval.reshape(1)
            constraint_eval.append(st_constraint_eval)
        # if self.m_st == 0 and self.m_det == 0:
        constraint_eval = torch.concatenate(constraint_eval)

        return constraint_eval
    


