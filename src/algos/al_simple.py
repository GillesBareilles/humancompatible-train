from typing import Tuple,Callable, List
import torch.utils.data.dataloader
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.optimize import linprog
from qpsolvers import solve_qp
import autoray as ar
import timeit
from itertools import cycle
from .utils import net_grads_to_tensor, net_params_to_tensor
import cvxpy as cp
import torch

m_det = 0
m_st = 2
import timeit
constr_sampling_interval = 1
max_runtime = 15

def one_sided_loss_constr(loss, net, c_data):
    w_inputs, w_labels = c_data[0]
    b_inputs, b_labels = c_data[1]
    w_outs = net(w_inputs)
    if w_labels.ndim == 0:
        w_labels = w_labels.reshape(1)
        b_labels = b_labels.reshape(1)
    else:
        w_labels = w_labels.unsqueeze(1)
        b_labels = b_labels.unsqueeze(1)
    w_loss = loss(w_outs, w_labels)
    b_outs = net(b_inputs)
    b_loss = loss(b_outs, b_labels)

    return w_loss - b_loss



def AugLagr(net: torch.nn.Module, data, w_ind, b_ind, batch_size, loss_bound, maxiter, max_runtime=np.inf,
            start_lambda=None,
            update_lambda=True):
        
    history = {'loss': [],
               'constr': [],
               'w': []}
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)    
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
    loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True))
    loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True))
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(2) if start_lambda is None else start_lambda
    c = torch.ones_like(_lambda)
    beta = 2.
    p = 2.
    
    run_start = timeit.default_timer()
    for iteration, data in enumerate(loader):
        net.zero_grad()
        
        current_time = timeit.default_timer()
        if max_runtime > 0 and current_time - run_start >= max_runtime:
            print(current_time - run_start)
            return
        
        w_sample = next(loader_w)
        b_sample = next(loader_b)
        
        inputs, labels = data
        outputs = net(inputs)
        if labels.dim() < outputs.dim():
            labels = labels.unsqueeze(1)
        loss_eval = loss_fn(outputs, labels)
        constraint_eval = torch.tensor([c1(net, [w_sample, b_sample]), 
                           c2(net, [w_sample, b_sample])])
        constraint_eval = torch.maximum(constraint_eval, torch.zeros(2))
        
        penalty_term = (1/p) * torch.sum(c * torch.abs(constraint_eval)**p)    # torch.norm(constraint_eval, p=_p)
        lag_term = _lambda @ constraint_eval
        L = loss_eval + lag_term + penalty_term
    
        L.backward()
        optimizer.step()        
        
        print(f'{iteration}|{loss_eval.detach().numpy()}|{_lambda.detach().numpy()}|{constraint_eval}|{c.detach().numpy()}', end='\r')
        
        for i in range(len(constraint_eval)):
            if iteration != 0 and constraint_eval[i] > 0.5*old_constraint_eval[i]:
                if update_lambda and _lambda[i] < 1e3:
                    _lambda[i] += c[i] * torch.abs(constraint_eval[i])**(p-1)
                if c[i] < 1e3:
                    c[i] *= 2
                
                
        old_constraint_eval = deepcopy(constraint_eval)
        history['w'].append(deepcopy(net.state_dict()))
        
    return history
    