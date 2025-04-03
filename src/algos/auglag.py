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
                lambda_bound = 1e3,
                pmult_bound = 1e3,
                start_lambda=None,
                update_lambda=True,
                update_pen = True,
                device='cpu'):
        
    history = {'loss': [],
               'constr': [],
               'w': []}
    
    c1 = lambda net, d: one_sided_loss_constr(loss_fn, net, d) - loss_bound
    c2 = lambda net, d: -one_sided_loss_constr(loss_fn, net, d) - loss_bound
    data_w = torch.utils.data.Subset(data, w_ind)
    data_b = torch.utils.data.Subset(data, b_ind)
    loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True, generator=torch.Generator(device=device))
    loader_w = cycle(torch.utils.data.DataLoader(data_w, batch_size, shuffle=True, generator=torch.Generator(device=device)))
    loader_b = cycle(torch.utils.data.DataLoader(data_b, batch_size, shuffle=True, generator=torch.Generator(device=device)))
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())
    n = sum(p.numel() for p in net.parameters())
    
    _lambda = torch.zeros(2) if start_lambda is None else start_lambda
    if update_pen:
        c = 1.
    else:
        c = 50
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
        constraint_eval = torch.tensor([
            c1(net, [w_sample, b_sample]), 
            c2(net, [w_sample, b_sample])
        ])
        constraint_eval = torch.maximum(constraint_eval, torch.zeros(2))
        
        penalty_term = (1/p) * torch.sum(c * torch.abs(constraint_eval)**p)    # torch.norm(constraint_eval, p=_p)
        lag_term = _lambda @ constraint_eval
        L = loss_eval + lag_term + penalty_term
    
        L.backward()
        optimizer.step()        
        
        print(f'{iteration}|{loss_eval.detach().cpu().numpy()}|{_lambda.detach().cpu().numpy()}|{constraint_eval}|{c}', end='\r')
        # print(f'{iteration}', end='\r')        
        # recalculate objective and constraint values based on updated network weights
        
        constraint_eval_updated = torch.tensor([c1(net, [w_sample, b_sample]), c2(net, [w_sample, b_sample])])
        constraint_eval_updated = torch.maximum(constraint_eval_updated, torch.zeros(2))
        # with torch.inference_mode():
        for i in range(len(constraint_eval_updated)):
            if iteration != 0 and constraint_eval_updated[i] > 0.5*constraint_eval[i]:
                lambda_upd = _lambda[i] + c * torch.abs(constraint_eval_updated[i])**(p-1)
                if update_lambda and lambda_upd < lambda_bound:
                    _lambda[i] = lambda_upd
            
        if c < pmult_bound:
            c *= 2    
                
                
        # old_constraint_eval = deepcopy(constraint_eval)
        history['w'].append(deepcopy(net.state_dict()))
        
    return history