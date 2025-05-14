from typing import Iterable, Tuple,Callable, List
import torch.utils.data.dataloader
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
import autoray as ar
import timeit
from itertools import cycle

from src.algorithms.c_utils.constraint import FairnessConstraint
from src.algorithms.c_utils.constraint_fns import *
from .utils import net_grads_to_tensor, net_params_to_tensor
import cvxpy as cp
import torch

m_det = 0
m_st = 2
import timeit
constr_sampling_interval = 1

def project(x, m):
    # for i in range(1,m+1):
    #     if x[-i] < 0:
    #         x[-i] = 0
    return x


def SwitchingSubgradient_pretty(
    net: torch.nn.Module,
    data: torch.utils.data.Dataset,
    constraints: Iterable[FairnessConstraint],
    ctol: float,
    f_stepsize_rule: str,
    f_stepsize: float,
    c_stepsize_rule: str,
    c_stepsize: float,
    batch_size: int = 8,
    epochs: int = 1,
    seed = 42,
    device ='cpu',
    max_runtime = np.inf,
    save_iter = None):
        
    history = {'loss': [],
               'constr': [],
               'w': [],
               'time': [],
               'n_samples': []}
    
    c = constraints

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_eval = None
    c_t = None
    run_start = timeit.default_timer()
    current_time = timeit.default_timer()
    
    f_eta_t = f_stepsize
    c_eta_t = c_stepsize
    
    f_iters = 0
    c_iters = 0
    for epoch in range(epochs):
        
        if current_time - run_start >= max_runtime:
            break
        gen = torch.Generator(device=device)
        gen.manual_seed(seed+epoch)
        loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=gen)
        
        for iteration, f_sample in enumerate(loader):
            current_time = timeit.default_timer()
            history['time'].append(current_time - run_start)
            if max_runtime > 0 and current_time - run_start >= max_runtime:
                print(current_time - run_start)
                break
            
            net.zero_grad()
            
            # generate sample of constraints
            c_sample = [ci.sample_loader() for ci in constraints]
            # calc constraints and update multipliers (line 3)
            with torch.no_grad():
                c_t = torch.concat([ci.eval(net, c_sample[i]).reshape(1) for i, ci in enumerate(c)])
                c_max = torch.max(c_t)
            history['constr'].append(c_max.cpu().detach().numpy())

            x_t = net_params_to_tensor(net, flatten=True, copy=True)
            
            if c_max >= ctol:
                c_iters += 1
            
                history['n_samples'].append(batch_size*2)
                # calculate grad on an independent sample
                c_sample = [ci.sample_loader() for ci in constraints]
                c_t2 = torch.concat([ci.eval(net, c_sample[i]).reshape(1) for i, ci in enumerate(c)])
                c_max2 = torch.max(c_t2)
                c_max2.backward()
                c_grad = net_grads_to_tensor(net)
                if c_stepsize_rule == 'adaptive':
                    c_eta_t = c_max / torch.norm(c_grad)**2
                elif c_stepsize_rule == 'const':
                    c_eta_t = c_stepsize
                elif c_stepsize_rule == 'dimin':
                    c_eta_t = c_stepsize / np.sqrt(c_iters)
                
                x_t1 = project(x_t - c_eta_t*c_grad, m=2)
            else:
            
                history['n_samples'].append(batch_size)
                f_iters += 1
                f_inputs, f_labels = f_sample
                outputs = net(f_inputs)
                if f_labels.dim() < outputs.dim():
                    f_labels = f_labels.unsqueeze(1)
                loss_eval = loss_fn(outputs, f_labels)
                loss_eval.backward()
                history['loss'].append(loss_eval.cpu().detach().numpy())
                f_grad = net_grads_to_tensor(net)
                
                if f_stepsize_rule == 'dimin':
                    f_eta_t = f_stepsize / np.sqrt(f_iters)
                elif f_stepsize_rule == 'const':
                    f_eta_t = f_stepsize
                x_t1 = project(x_t - f_eta_t*f_grad, m=2)
            
            start = 0
            with torch.no_grad():
                w = net_params_to_tensor(net, flatten=False, copy=False)
                for i in range(len(w)):
                    end = start + w[i].numel()
                    w[i].set_(x_t1[start:end].reshape(w[i].shape))
                    start = end
            
            if loss_eval is not None and c_t is not None:
                with np.printoptions(precision=6, suppress=True):
                    print(f'{epoch:2} | {iteration:5}|{loss_eval.detach().cpu().numpy()}|{c_t.detach().cpu().numpy()}', end='\r')
            history['w'].append(deepcopy(net.state_dict()))
        
    ######################
    ### POSTPROCESSING ###    
    ######################
    
    if not (save_iter is None):
        model_ind = np.random.default_rng(seed=seed).integers(save_iter,len(history['w']))
        net.load_state_dict(history['w'][model_ind])
        history['w'].append(deepcopy(net.state_dict()))
    
    print('\n')
    print(c_iters)
    return history